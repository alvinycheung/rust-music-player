# Audio Visualizer (Nannou)

An immersive audio visualization experience built with Rust and the Nannou creative coding framework.

## Features

- Real-time audio spectrum analysis using FFT
- Dynamic 3D Merkaba geometry that responds to music
- Starfield background with audio-reactive movement
- Interactive spectrum visualization with mouse influence
- Music player with file browser and playback controls
- Volume control with settings persistence

## Getting Started

### Prerequisites

- Rust toolchain (install from [rustup.rs](https://rustup.rs))
- Audio input device (microphone or line-in)
- Music files in MP3 or other supported formats

### Running the Visualizer

```bash
# Run in fullscreen mode (recommended)
cargo run --release --bin nannou-visualizer -- --fullscreen

# Run in windowed mode
cargo run --release --bin nannou-visualizer
```

## Controls

- **Space**: Play/pause current song
- **N**: Next song
- **P**: Previous song
- **M**: Toggle song browser
- **F**: Toggle FPS display
- **H**: Hide/show UI elements
- **Up/Down Arrow**: Adjust volume
- **Mouse Movement**: Influence visualization dynamics
- **Mouse Wheel**: Scroll through song list (when browser is open)
- **Mouse Click**: Select song in browser
- **Double Click**: Play selected song immediately
- **Esc**: Exit application

## Technical Details

The visualizer features several components:

- **Audio Processing**: Real-time FFT analysis of audio data using rustfft
- **SpectrumVisualizer**: Dynamic circular spectrum analyzer with history trails
- **Merkaba**: 3D geometry that transforms based on audio frequencies
- **Starfield**: Background stars that move and pulse with the music
- **UI System**: Song browser and playback controls with interactive elements

## Performance Notes

The application is optimized for smooth performance:

- GPU-accelerated rendering via Nannou/wgpu
- Efficient audio processing with minimal latency
- Adaptive animation speed based on system performance
- Multi-threaded audio processing to keep the UI responsive

## Music Setup

Place music files in the `src/music` directory. Supported formats include:

- MP3
- WAV
- FLAC
- OGG

The visualizer will automatically detect and list all available songs.

## Customization

Modify constants at the top of `src/nannou_visualizer.rs` to adjust visualization parameters:

- `FFT_SIZE`: Resolution of frequency analysis
- `MAX_AMPLITUDE`: Maximum visual amplitude for spectrum display
- `TRACER_LENGTH`: Number of frames to keep in animation history
- `MAX_STARS`: Density of the starfield background

## License

MIT
