# PEACOCK DETECTOR AI

[➡️ **Live Demo on Hugging Face Spaces**](https://huggingface.co/spaces/mr-sachin/Peacock_Detection_with_YOLO)

Smart, real‑time peacock detection and tracking system built with YOLO, Flask, and a modern web dashboard. Designed to help farmers monitor fields, understand peacock movement patterns, and reduce crop damage with less manual effort.

---

## Problem

In many regions, peacocks are a serious challenge for farmers:

- **Crop damage:** Peacocks feed on young shoots, grains, and fruits, damaging a wide variety of crops (paddy, pulses, vegetables, etc.). Even a small group can destroy a large area in a short time.
- **Manual monitoring is exhausting:** Farmers often guard fields early morning and evening, walking long boundaries and visually scanning for birds. This is **time‑consuming, physically tiring, and hard to sustain** every single day across large or multiple fields.
- **No continuous visibility:** Once a farmer leaves the field, there is no clear way to know when and where peacocks entered, how long they stayed, or how many birds visited. This makes it difficult to plan protection measures or understand patterns.
- **Reactive, not proactive:** Most actions are taken **after** visible damage happens. Without timely alerts or visual evidence, farmers struggle to prove and justify losses or optimize scare strategies (sprinklers, sound devices, etc.).

There is a clear need for an automated, always‑on visual system that can **detect peacocks in real time**, highlight which area of the field they are entering, and reduce the farmer’s manual workload.

---

## Abstract

**Peacock Detector AI** is an end‑to‑end computer vision solution that uses a custom‑trained YOLO model to automatically detect and track peacocks in video streams and images.

The system consists of:

- A **YOLO-based detection engine** fine‑tuned specifically on peacock imagery to filter out other animals and background noise.
- A **Flask-backed web dashboard** that accepts images, uploaded videos, or live camera streams and overlays high‑confidence peacock detections with bounding boxes and IDs.
- A **visual “sprinkler zone” interface** that divides the camera view into four regions, showing which quadrant is currently occupied by peacocks using animated circular sprinkler icons and status text. This makes it easy to understand where peacocks are entering without reading raw coordinates.
- A **lite / full mode design**, allowing a lightweight single‑image mode suitable for CPU-only environments (like Hugging Face Spaces free tier), while keeping full video + realtime tracking available for local or higher‑capacity deployments.

By automating peacock detection and providing a clear, visual field layout, Peacock Detector AI aims to reduce manual field patrols, give farmers timely awareness of intrusions, and lay the foundation for future automated deterrent systems (sprinklers, sound devices, or lights) that can be triggered directly from detection events.

---

## How the Model Was Built

**End-to-end training pipeline (high level)**

⬜ Pretrained YOLO11n (COCO weights)
	⬇ fine-tune
⬛ Roboflow peacock dataset (multi-object, peacock labeled)
	⬇ produces
🟦 Base detector that can detect general objects (not yet focused only on peacocks)
	⬇ auto-labels
🟩 2,000 peacock-only images from CV dataset
	⬇ plus
🟩 100 web-scraped peacock images (open-source) manually + auto-labeled
	⬇ combine
🟨 Merged peacock dataset (3 sources together)
	⬇ train 20 epochs on T4 GPU
🟪 Final YOLO11n peacock-only model for real-time video

**Detailed steps**

- **Step 1 – Start from pretrained YOLO11n:**
	- Loaded the official pretrained YOLO11n model as the base backbone and head.
- **Step 2 – Fine-tune on Roboflow peacock dataset:**
	- Used a Roboflow peacock dataset containing multiple objects with peacock labels.
	- Trained a **base detector** that can detect objects in general, but is not yet specialized only for peacocks.
- **Step 3 – Auto-label peacock-only datasets:**
	- Took a CV dataset of **~2,000 peacock-only images**.
	- Web-scraped an additional **~100 open-source peacock images** and labeled them (manually + using the base model for auto-labeling assistance).
	- Used the base detector to speed up annotation and clean labels where needed.
- **Step 4 – Merge and train final model:**
	- Combined all three sources (Roboflow multi-object peacock dataset + CV peacock-only dataset + web-scraped peacock images).
	- Trained the YOLO11n model for **20 epochs on a T4 GPU**, focusing on the single "peacock" class.
- **Step 5 – Deploy for real-time detection:**
	- Exported the best checkpoint and integrated it into this app as `model/best_peacock_model_v2.pt`.
	- The resulting model is specialized to **detect only peacocks in real-time video and images**, filtering out other objects.

---

## 📁 Project Structure

```text
Peacock_Detection_with_YOLO/
├── app.py              # Flask backend API (YOLO inference, routes)
├── index.html          # Frontend dashboard (camera, video, sprinklers UI)
├── requirements.txt    # Python dependencies
├── data/               # Placeholder for input videos/images (local use)
└── model/
    └── best_peacock_model_v3.pt  # Trained YOLO11n peacock-only weights
```

---

## 🖥️ **Run It Locally**

Follow these steps to download and run the app on your own machine:

1. **Clone the repository**
  - `git clone https://github.com/Sachin-MR05/Peacock_Detection_with_YOLO.git`
  - `cd Peacock_Detection_with_YOLO`

2. **Create and activate a virtual environment** (recommended)
  - Windows (PowerShell):
    - `python -m venv .venv`
    - `./.venv/Scripts/Activate.ps1`

3. **Install dependencies**
  - `pip install -r requirements.txt`

4. **Set app mode to full (optional but recommended for local)**
  - Create a `.env` file in the project root with:
    - `APP_MODE=full`

5. **Run the Flask app**
  - `python app.py`
  - The server will start on: `http://localhost:5000`

6. **Open the dashboard**
  - In your browser, open `http://localhost:5000`
  - Use the UI to upload an image/video or start the camera and see real-time peacock detection with the sprinkler visualization.

---

## 🤝 **Contributing**

We welcome contributions from everyone! Whether you're fixing bugs, adding features, or improving documentation, your help makes ChangeDetection better.

### **How to Contribute**

#### **1. Report Issues & Request Features**
Found a bug or want a new feature? Open an issue on GitHub:

📌 **Steps:**
1. Go to [GitHub Issues](https://github.com/your-repo/issues)
2. Click "New Issue"
3. Provide clear description:
   - **Bug Report**: Describe what went wrong, steps to reproduce, expected behavior
   - **Feature Request**: Explain the feature and why it would be valuable
   - **Improvement Suggestion**: Point out areas for enhancement


### **Recognition & Encouragement**

⭐ **Star this repository** to show your support and help others discover ChangeDetection!


---

## 📋 **Development Roadmap** (Planned Features)

### **Phase 1: Model Enhancement** 🔧
- Improve robustness across lighting conditions and camera angles.
- Experiment with additional data augmentation and longer training.

### **Phase 2: Edge Device Integration** 🛰️
- Package the peacock-only model for low-power edge devices (Jetson, Raspberry Pi, etc.).
- Optimize inference for real-time performance directly at the field edge.

### **Phase 3: AI Agent Automation** 🤖
- Connect AI agents to monitor detections and trigger actions automatically.
- Integrate with sprinkler controllers, sound devices, or alerts for fully automated deterrence.

---

## **Acknowledgments**

- **Farmers facing peacock issues:** This project is inspired by farmers who regularly lose crops to peacock activity and need practical, affordable field monitoring tools.
- **Dataset curation:** Thanks to the open peacock datasets, Roboflow resources, and carefully web‑scraped imagery that made it possible to build a focused peacock detector.
- **Model ecosystem:** Appreciation to the YOLO community and open‑source contributors whose pretrained models and tooling enabled fast experimentation and fine‑tuning for this use case.
---
## 👨‍💻 **About the Creator**

<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Poppins&weight=600&size=24&duration=2800&pause=900&color=0F766E&center=true&vCenter=true&width=720&lines=AI+%26+Machine+Learning+Enthusiast;Building+Intelligent+Systems+for+Real-World+Impact" alt="Sachin typing introduction" />
</div>

<p align="center">
  <strong>SACHIN M R</strong> - Passionate AI &amp; Machine Learning Enthusiast
</p>

<p align="center">
  I am dedicated to harnessing the power of <strong>Artificial Intelligence</strong> to make people's lives easier and enable autonomous systems across every field. My journey involves deep learning, machine learning, and AI agents.
</p>

<div align="center">

### **Current Focus**

</div>

<p align="center">
  📚 <strong>Learning &amp; mastering Deep Learning architectures</strong><br>
  🤖 <strong>Building AI Agents with advanced reasoning capabilities</strong><br>
  🌍 <strong>Creating autonomous systems for real-world problems</strong><br>
  🛰️ <strong>Satellite imagery analysis &amp; geospatial AI applications</strong>
</p>

<div align="center">

### **Connect & Follow**

  <a href="https://www.linkedin.com/in/mr-sachin">
    <img src="https://img.shields.io/badge/LinkedIn-mr--sachin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://github.com/Sachin-MR05">
    <img src="https://img.shields.io/badge/GitHub-Sachin--MR05-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://huggingface.co/mr-sachin">
    <img src="https://img.shields.io/badge/Hugging%20Face-mr--sachin-FFB000?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  </a>

</div>

<p align="center">
  Eager to connect for collaborations, internships, and meaningful technical discussions.
</p>

---

<div align="center">

### **Made with ❤️ by Sachin**

*Empowering autonomous systems for a better future*

</div>