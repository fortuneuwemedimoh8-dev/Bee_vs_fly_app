# 🐝 vs 🪰 Bee vs Fly Image Classifier

**CE 11 Mini Project**

A Streamlit web app that uses a fine-tuned deep learning model to classify uploaded images as either a **bee** or a **fly**, along with a confidence score.

🔗 **Live App:** [Click here to try it](https://beevsflyapp-gqzyancaqsmtn32rquhbvq.streamlit.app/)

---

## 👥 Team Members & Roles
| Fortune Uwemedimoh | [@fortuneuwemedimoh8-dev](https://github.com/fortuneuwemedimoh8-dev) | Model development, training, deployment |
| Oswin | [@Oswin727](https://github.com/Oswin727) | Co-development support, testing, debugging assistance |
| Baker | [@Baker1375](https://github.com/Baker1375) | Dataset research & documentation support |
| Favour Effiong | [@favoureffiong995-hash](https://github.com/favoureffiong995-hash) | Testing |
| Ini Ernest | [@Ini-ernest77](https://github.com/Ini-ernest77) | Dataset research & testing |
---

## 📌 Project Overview

This project builds an image classifier to distinguish between bees and flies using transfer learning. The pipeline covers dataset preparation, model training, and web deployment.

## 🛠️ How It Was Built

1. **Dataset** — Sourced from the [Insects Image Dataset](https://www.kaggle.com/datasets/ismail703/insects) on Kaggle, filtered down to the Bee and Fly classes.
2. **Data Cleaning** — Removed corrupted/unreadable image files using TensorFlow's image decoder.
3. **Train/Validation Split** — 80/20 split using `split-folders`.
4. **Model** — Transfer learning with **MobileNetV2** (pretrained on ImageNet), with a custom classification head (Global Average Pooling → Dropout → Dense sigmoid output).
5. **Training** — 10 epochs, achieving ~99% training accuracy and ~99% validation accuracy.
6. **Deployment** — Packaged as a `.keras` model and deployed via **Streamlit Cloud**, connected to this GitHub repo.

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Google Colab (training environment)
- GitHub + Streamlit Cloud (deployment)

## 🚀 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
