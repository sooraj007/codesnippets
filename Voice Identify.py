
#imports above
# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class VoiceRecognitionModel(nn.Module):
    def __init__(self, num_speakers=2):
        """
        A multi-speaker model with `num_speakers` classes.
        If you start with zero enrolled speakers, we set up a placeholder,
        but the final layer will be resized once you enroll the first speaker.
        """
        super().__init__()
        # Updated input channels from 101 to 105 to match our feature dimensions
        self.conv1 = nn.Conv1d(105, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.gru = nn.GRU(256, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_speakers)  # final classifier
        self.dropout = nn.Dropout(0.5)

        # Adjusted loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)
        
        # Add scheduler for learning rate
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def forward(self, x):
        """
        x: [batch_size, 105, time_frames]
        Returns: [batch_size, num_speakers]
        """
        x = self.dropout(torch.relu(self.conv1(x)))   # -> [batch, 64, T/2 approx]
        x = self.dropout(torch.relu(self.conv2(x)))   # -> [batch, 128, T/4 approx]
        x = self.dropout(torch.relu(self.conv3(x)))   # -> [batch, 256, T/8 approx]
        x = x.transpose(1, 2)                         # -> [batch, T/8, 256]
        x, _ = self.gru(x)                            # -> [batch, T/8, 256] (bidirectional = 2*128=256)
        x = x.mean(dim=1)                             # average over time
        x = self.fc(x)                                # -> [batch, num_speakers]
        
        # Higher temperature for more conservative predictions
        temperature = 2.0
        return torch.softmax(x / temperature, dim=1)

class VoiceRecognitionSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000
        
        # Initialize pyannote embedding model with more detailed error handling
        try:
            print("Attempting to load pyannote model...")
            from pyannote.audio import Model
            model = Model.from_pretrained("pyannote/embedding",
                                        use_auth_token=os.environ.get("HF_TOKEN"))
            
            self.inference = Inference(model, window="whole")
            if torch.cuda.is_available():
                self.inference.to(self.device)
            
            print("Successfully loaded pyannote model")
        except Exception as e:
            print(f"Error loading pyannote model: {str(e)}")
            print(f"HF_TOKEN present: {'HF_TOKEN' in os.environ}")
            self.inference = None
            
        # Speaker database: name -> embedding
        self.speaker_db = {}
        
        # Load existing speakers if any
        self.load_system_state()

    def save_system_state(self):
        """Saves speaker embeddings and metadata"""
        try:
            save_dir = Path("voice_profiles")
            save_dir.mkdir(exist_ok=True)

            # Save speaker embeddings
            for name, embedding in self.speaker_db.items():
                torch.save({
                    'embedding': embedding,
                    'name': name
                }, save_dir / f"{name}.pt")

            print(f"Saved {len(self.speaker_db)} speaker profiles")

        except Exception as e:
            print(f"Error saving system state: {str(e)}")

    def load_system_state(self):
        """Loads saved speaker embeddings"""
        save_dir = Path("voice_profiles")
        if not save_dir.exists():
            return

        for file in save_dir.glob("*.pt"):
            try:
                data = torch.load(file)
                name = data['name']
                embedding = data['embedding']
                self.speaker_db[name] = embedding
                print(f"Loaded profile for speaker: {name}")
            except Exception as e:
                print(f"Error loading {file}: {e}")

    def enroll_speaker(self, audio, name):
        """Enrolls a new speaker using pyannote embeddings"""
        if not audio or self.inference is None:
            return "No audio provided or model not loaded"

        try:
            # Convert audio to numpy array if needed
            if isinstance(audio, tuple):
                audio = audio[1]
            
            # Check audio duration (minimum 2 seconds recommended)
            duration = len(audio) / self.sample_rate
            if duration < 2.0:
                return f"Audio too short ({duration:.1f}s). Please provide at least 2 seconds of audio."
            
            # Convert to float32 and ensure correct shape
            audio = torch.tensor(audio, dtype=torch.float32)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)  # Add channel dimension
            
            # Normalize audio
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
            
            # Get embedding using pyannote
            embedding = self.inference({
                "waveform": audio,
                "sample_rate": self.sample_rate
            })
            
            # Store the embedding
            self.speaker_db[name] = embedding
            
            # Save the updated system state
            self.save_system_state()
            
            return f"Successfully enrolled {name} (audio duration: {duration:.1f}s)"
            
        except Exception as e:
            return f"Error during enrollment: {str(e)}"

    def identify_speaker(self, audio):
        """Identifies speaker using cosine similarity"""
        if not audio or self.inference is None:
            return "No audio provided or model not loaded"

        if not self.speaker_db:
            return "No speakers enrolled yet"

        try:
            # Convert audio to numpy array if needed
            if isinstance(audio, tuple):
                audio = audio[1]
            
            # Check audio duration
            duration = len(audio) / self.sample_rate
            if duration < 2.0:
                return f"Audio too short ({duration:.1f}s). Please provide at least 2 seconds of audio."
            
            # Format audio according to pyannote requirements
            if isinstance(audio, np.ndarray):
                waveform = torch.from_numpy(audio).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                
                audio_input = {
                    "waveform": waveform,
                    "sample_rate": self.sample_rate
                }
            else:
                return f"Unexpected audio type: {type(audio)}"
            
            # Get embedding and ensure correct shape
            test_embed = self.inference(audio_input)
            test_embed = torch.from_numpy(test_embed).float()
            
            # Compare with all stored embeddings
            best_speaker = None
            best_score = float("-inf")
            
            results = []
            for name, spk_embed in self.speaker_db.items():
                spk_embed = torch.from_numpy(spk_embed).float()
                # Ensure both tensors are 2D for cosine similarity
                similarity = F.cosine_similarity(
                    test_embed.view(1, -1),  # reshape to [1, features]
                    spk_embed.view(1, -1)    # reshape to [1, features]
                ).item()
                
                results.append((name, similarity))
                if similarity > best_score:
                    best_score = similarity
                    best_speaker = name

            # Format detailed results
            details = "\n".join(f"{name}: {score:.3f}" for name, score in results)
            
            # Use threshold for confidence (note: now using direct similarity, not distance)
            if best_score < 0.7:
                return f"Detailed Scores:\n{details}\n\nUnknown speaker (similarity={best_score:.3f})"
            else:
                return f"Detailed Scores:\n{details}\n\nIdentified as {best_speaker} (similarity={best_score:.3f})"

        except Exception as e:
            return f"Error during identification: {str(e)}"

    def load_previous_audio(self, audio_path):
        """
        Helper method to safely load previous speaker audio data
        """
        try:
            data = torch.load(audio_path, weights_only=True)
            if isinstance(data, dict) and 'tensor' in data:
                return data['tensor'].to(self.device)
            elif isinstance(data, torch.Tensor):
                return data.to(self.device)
            return None
        except Exception as e:
            print(f"Error loading audio data: {e}")
            return None

    def _normalize_length(self, features, target_length=2000):
        """
        Normalize the length of feature vectors to a fixed size
        Args:
            features: tensor of shape [channels, time]
            target_length: desired length of time dimension
        Returns:
            tensor of shape [channels, target_length]
        """
        current_length = features.shape[1]
        
        if current_length == target_length:
            return features
        
        # If shorter, pad with zeros
        if current_length < target_length:
            padding = torch.zeros(
                (features.shape[0], target_length - current_length),
                device=features.device
            )
            return torch.cat([features, padding], dim=1)
        
        # If longer, interpolate
        return torch.nn.functional.interpolate(
            features.unsqueeze(0),
            size=target_length,
            mode='linear'
        ).squeeze(0)

############################################################
#            GRADIO INTERFACE
############################################################
def create_gradio_interface():
    system = VoiceRecognitionSystem()

    with gr.Blocks(title="Multi-Speaker Voice Recognition System") as interface:
        gr.Markdown("## Multi-Speaker Voice Recognition System (Improved)")

        with gr.Tab("Enroll New Speaker"):
            with gr.Row():
                with gr.Column():
                    audio_mic = gr.Audio(source="microphone", type="numpy", label="Record Audio (minimum 2s)")
                with gr.Column():
                    audio_file = gr.Audio(source="upload", type="numpy", label="Upload Audio File (minimum 2s)")
            
            name_input = gr.Textbox(label="Speaker Name", value="sooraj")
            enroll_btn_mic = gr.Button("Enroll (Microphone)")
            enroll_btn_file = gr.Button("Enroll (File)")
            enroll_output = gr.Textbox(label="Enrollment Status")

            enroll_btn_mic.click(
                fn=system.enroll_speaker,
                inputs=[audio_mic, name_input],
                outputs=enroll_output
            )
            enroll_btn_file.click(
                fn=system.enroll_speaker,
                inputs=[audio_file, name_input],
                outputs=enroll_output
            )

        with gr.Tab("Identify Speaker"):
            with gr.Row():
                with gr.Column():
                    mic_audio2 = gr.Audio(source="microphone", type="numpy", label="Record Audio (>=3s)")
                with gr.Column():
                    file_audio2 = gr.Audio(source="upload", type="numpy", label="Upload Audio File (>=3s)")

            identify_btn_mic = gr.Button("Identify (Microphone)")
            identify_btn_file = gr.Button("Identify (File)")
            identify_output = gr.Textbox(label="Recognition Result")

            identify_btn_mic.click(
                fn=system.identify_speaker,
                inputs=mic_audio2,
                outputs=identify_output
            )
            identify_btn_file.click(
                fn=system.identify_speaker,
                inputs=file_audio2,
                outputs=identify_output
            )

    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
