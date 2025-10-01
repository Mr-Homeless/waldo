# Visual Cheat Detection System (Waldo_alpha_01)

**Advanced AI-powered cheat detection for Counter-Strike 2 gameplay footage**

This system uses cutting-edge Vision Transformer deep learning technology to analyze gameplay clips and detect potential cheating behavior from the footage. The current version is specific to CS2 and must use a user-trained model. 

---

## What This Does

- **Analyzes CS2 gameplay videos** - Upload your raw gameplay footage or individual clips
- **Focuses on killshots** - Automatically extracts 2-second clips by detecting headshots from gameplay audio
- **Trains locally from your footage** - Select your processed clips folder and train or fine-tune a model with labels (cheating) or (not cheating) 
- **Analyzes processed clips** - Provides detailed confidence scores from 0.0 (likely legitimate) to 1.0 (likely cheating) on clips 
- **Frame-by-frame analysis** - See exactly which 16 frames the model analyzed for each clip
- **Persistent results** - All analysis results are automatically saved and can be revisited later
- **Export capabilities** - Export results as JSON for further analysis

## 🚀 Quick Start Guide (Windows 11)

### Step 1: Download and Setup

1. **Download this project** to your computer (extract the ZIP)

2. **Run wsl-install.bat as admin** (not a virus trust me bro)
    Right click wsl-install.bat and click run as administrator. This will take a few minutes and will look frozen, but it's working in the background to install WSL. 

    Once the terminal auto-closes, reboot your PC and run wsl-install.bat as admin again and follow the prompts to set up a username and password. 

3. **Run wsl-setup.bat NOT as admin** 
    Double click wsl-setup.bat to run it without elevated privilages. This will auto-install the conda environment and all the requirements within the new WSL you just made. 


### Step 2: Prepare model for training/inferencing

1. **Download the model**
    Go to https://huggingface.co/jinggu/jing-model/blob/main/vit_g_ps14_ak_ft_ckpt_7_clean.pth and download the .pth file

2. **Place the model**

    Place this .pth file in the "deepcheat/VideoMAEv2" folder found in the extracted project files. Paste any other downloaded models in "deepcheat/VideoMAEv2/output" for inferencing or fine tuning. 

3. **You're ready to train a model and analyze CS2 clips!**


### Step 3: Run the Application

1. **Start the server**:
    Double click wsl-run.bat

2. **Open your web browser** and go to:
    http://localhost:5000

3. **Follow the steps** to train a model and analyze footage


## 🚀 Quick Start Guide (Linux)


1. **Paste this** in your terminal - Script isn't currently tested. If it doesn't work, make install.sh executable and run it. Then do the same for run.sh

```bash
echo "🚀 Starting Waldo installation..."

# Clone the repository
echo "📥 Downloading project from GitHub..."
git clone https://github.com/Mr-Homeless/waldo.git
cd waldo

# Make install.sh executable and run it
echo "🔧 Installing dependencies..."
chmod +x install.sh
./install.sh

# Download the model file
echo "🤖 Downloading AI model (1.9GB - this will take a few minutes)..."
wget --show-progress -O deepcheat/vit_g_ps14_ak_ft_ckpt_7_clean.pth \
  https://huggingface.co/jinggu/jing-model/resolve/main/vit_g_ps14_ak_ft_ckpt_7_clean.pth

# Ask user if they want to run the server
echo ""
echo "✅ Installation complete!"
echo ""
read -p "Would you like to start the server now? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🌐 Starting server on http://localhost:5000"
    chmod +x run.sh
    ./run.sh
else
    echo "To start the server later, run: cd waldo&& ./run.sh"
fi
```




---

## 📖 How to Use

### 1. Process Raw Footage
- **Upload CS2 gameplay videos** (any format: MP4, AVI, etc.)
- The system automatically detects kill moments using audio analysis
- Extracts 2-second clips around each detected kill
- Any resolution should work, but 59.94 or 60p footage is recommended
- Currently you must remove spectator kills and proximity headshot kills manually

### 2. Train or Fine-Tune Models on Processed Clips
- **Label your clips** as "Cheater" or "Not Cheater" with the drop down selector
- **Train new models** or fine-tune existing ones with your clips (currently you need at least 48 processed clips to start training)
- **Real-time progress tracking** with detailed json logs
- As Mr. Homeless trains more accurate models, they will be made public here for free download https://www.patreon.com/basicallyhomeless and can be placed in the deepcheat/output/ folder for use

### 3. Test Clips for Cheating
- **Select clips to analyze** (from footage processed in step 1.)
- **Choose a trained model** for analysis
- **Get detailed results** including:
  - Confidence scores
  - Probability estimates using sigmoid transformation
  - Frame-by-frame analysis showing exactly what frames and ROI the model examined
  - Color-coded confidence categories

### 4. View Detailed Results
- **Comprehensive dashboard** with statistics and visualizations
- **Individual clip analysis** with video playback
- **Frame viewer** showing the exact 16 frames analyzed
- **Export capabilities** for sharing or further analysis

---

## 🔧 Technical Features

### Advanced Analysis
- **Vision Transformer architecture** with 1 billion parameters
- **Adaptive resolution scaling** - works on any video resolution (1080p, 1440p, 4K, etc.)
- **Smart frame selection** - analyzes frames 85-100 where killshots occur
- **Center crop focus** - concentrates on the crosshair area 
- **Clip generator functionality** - Makes two second clips based on a sample sound -- change the sample sound to any "kill" sound effect for different games

### Smart Video Processing
- **Automatic kill detection** using audio pattern recognition
- **Resolution-adaptive cropping** maintains consistent field of view across different video qualities
- **Optimal frame extraction** targets the exact moments where cheating behavior is most visible

### Professional Results
- **Statistical analysis** with mean, median, standard deviation
- **Distribution visualization** showing score patterns
- **Confidence categorization** with clear color coding
- **Persistent storage** - results saved automatically for future reference

---

## 📊 Understanding Results

### Confidence Scores
- **0.8 - 1.0**: 🔴 **Very High Confidence - Likely Cheating**
- **0.6 - 0.8**: 🟠 **High Confidence - Possible Cheating**
- **0.4 - 0.6**: 🟡 **Medium Confidence - Uncertain**
- **0.2 - 0.4**: 🟢 **Low Confidence - Likely Legitimate**
- **0.0 - 0.2**: ✅ **Very Low Confidence - Likely Legitimate**

### What the AI Analyzes
The system examines:
- **Crosshair movement patterns** during engagements
- **Reaction timing** to enemy appearances
- **Tracking smoothness** and micro-corrections
- **Pre-aim positioning** before enemies are visible
- **Flick accuracy** and consistency patterns

---

## 🎥 Supported Video Formats

- **Resolution**: Any (1080p, 1440p, 4K, ultrawide, etc.)
- **Formats**: MP4, AVI, MOV, MKV, and most common video formats
- **Frame rates**: 59.94, 60fps, constant frame rate recommended 
- **Best quality**: Higher resolution and frame rate = better analysis

---

## ⚡ Performance Tips

### For Best Results:
1. **Use high-quality footage** (1440p+ recommended)
2. **Include multiple kills** in your uploaded videos
3. **Clear audio** helps with automatic kill detection
4. **Consistent crosshair placement** in center of screen

### System Requirements:
- **CPU**: 12th gen intel or Ryzen 5000 or newer
- **RAM**: 16GB+ recommended (32GB+ for large videos)
- **GPU**: NVIDIA GPU with CUDA support (other GPUs may work in compatibility mode but have not been tested)
- **Storage**: ~10GB free space for models and temporary files

---

## 🔒 Privacy & Data

- **All processing is local** - your footage never leaves your computer
- **No internet required** for analysis (only for initial setup)
- **Results stored locally** in the `evaluation_results` folder
- **Your data remains private** and under your control - no cloud connection needed

---

## 🛠️ Troubleshooting

- After training a new model or fine tuning a model, it may give an error code, but if it ran through the Epocs and trained, it did complete and the errors are likely not crucial to functioning. 

- If running this on Windows, it will run slightly slower and sometimes look frozen/won't have output. This is normal right now.

- Auto-clipping function Works for multiple audio streams, so if its not giving you 2 second clips, double check that the game audio is in the file and in-sync with the footage.



---

## 📜 License & Disclaimer

**This tool is for educational and analytical purposes.**

- Results should be considered **guidance, not definitive proof**
- Respect privacy and competitive integrity guidelines
- Use responsibly within gaming community standards
- This **is not the final, trained and tuned version of Waldo that will definitively tell you who is cheating**


---

## 🌟 Version Information

**Current Version**: Alpha 1.0
**Model Version**: VideoMAE v2 with CS2-specific training
**Last Updated**: September 19th, 2025

### TO-DO LIST:
- ☐ Create an OBS plugin that uses replay recording to record two second clips at the detected sound byte
- ☐ The wsl-setup.bat can be modified to put temp files in a better location and potentially speed up the training for Windows users
- ☐ Update pytorch without breaking everything - this should make training on newer GPUs faster
- ☐ Create a filter that deletes/removes clips from spectator view kills and proximity kills
- ☐ Have a whale lan with 500 locked down PCs to get massive amounts of clean labeled data 
- ☐ Train the terminator waldo model in year 2027 on RTX 6090Tis 

---

## 🤝 Support

For questions, issues, or feedback about this beta version, please provide:
- Your system specifications
- Error messages (if any)
- Screenshots of issues
- Description of what you were trying to do

**Remember**: This is not finalized software. While thoroughly tested, you may encounter issues. Your feedback helps improve the system for everyone!

---

**Happy training and analyzing! 🎯**
