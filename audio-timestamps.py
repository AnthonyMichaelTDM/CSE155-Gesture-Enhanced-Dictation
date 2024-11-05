import stable_whisper

# Load a model, you can choose between 'tiny', 'base', 'small', 'medium', and 'large'
model = stable_whisper.load_model('base')

# Path to your audio file (e.g., "audio.wav")
#audio_path = r"C:\\Users\\Fused\\OneDrive\\Desktop\\School\\UCMerced\\Fall 2024\\CSE 155\\Speech-to-Text\\harvard.wav"
audio_path = r"harvard.wav"


# regions on the waveform colored red are where it will likely be suppressed and marked as silent
# [q_levels]=20 and [k_size]=5 (default)
# stable_whisper.visualize_suppression('harvard.wav', 'image.png', q_levels=20, k_size = 5) 

# Run the transcription with stable timestamps
result = model.transcribe(audio_path)

# Print the transcription
print(type(result))

# Print each word with its corresponding timestamp
for segment in result.segments:
    for word_info in segment.words:
        word = word_info.word       # Accessing 'word' attribute
        start_time = word_info.start  # Accessing 'start' attribute
        end_time = word_info.end      # Accessing 'end' attribute
        print(f"Word: '{word}', Start Time: {start_time:.2f}s, End Time: {end_time:.2f}s")

# Print the structure of `word_info` to identify its attributes
# for segment in result.segments:
#     for word_info in segment.words:
#         print(word_info)  # Print out the word_info object to inspect its structure