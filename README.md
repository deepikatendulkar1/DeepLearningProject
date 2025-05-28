# 🧠 Automated Image Captioning Using Deep Learning

This project explores deep learning models for automated image captioning—a task that sits at the intersection of computer vision and natural language processing. We evaluate and compare the performance of several models, including BLIP and BLIP-2 variants, using the COCO 2017 dataset. Our findings highlight the effectiveness of fine-tuning and model architecture in producing accurate, fluent, and semantically aligned captions.

## 👩‍💻 Authors
- **Deepika Hemant Tendulkar** - George Mason University  
- **Sanjana Vegesna** - George Mason University  

## 🧪 Project Highlights
- Models: `BLIP`, `BLIP Fine-Tuned`, `BLIP-2 OPT 2.7B`, `BLIP-2 Flan-T5-XL`
- Dataset: COCO 2017 (10K train, 2K val samples used)
- Framework: Python, PyTorch, HuggingFace Transformers
- Evaluation: BLEU, METEOR, ROUGE-L, CIDEr, SPICE, CLIPScore
- Results: Fine-tuned BLIP and BLIP-2 OPT 2.7B outperform zero-shot baselines

## 📁 Project Structure
```
.
├── README.md
├── automated-image-captioning-coco.ipynb     # Full implementation
├── automated-image-captioning-report.pdf     # Final paper/report
├── requirements.txt                          # Python dependencies (optional)
└── /results                                   # Output evaluations, images, etc.
```

## 🖼️ Dataset
**COCO 2017**: A widely-used benchmark with over 118,000 training images and 5,000 validation images, each with five human-annotated captions.  
We used subsets for training and evaluation due to computational constraints.

## ⚙️ Models & Methods
### BLIP (Zero-Shot)
Vision Transformer (ViT) + BERT decoder. No fine-tuning; direct inference.

### BLIP (Fine-Tuned on COCO)
Same architecture fine-tuned on COCO captions to improve dataset-specific accuracy.

### BLIP-2 OPT 2.7B
Zero-shot captioning using a querying transformer + large language model (OPT-2.7B).

### BLIP-2 Flan-T5-XL
Zero-shot captioning using Flan-T5-XL as the language decoder for greater fluency.

## 📈 Evaluation Metrics
| Model              | BLEU-4 | METEOR | CIDEr | SPICE | CLIPScore |
|-------------------|--------|--------|-------|--------|------------|
| BLIP (Zero-Shot)  | 0.313  | 0.241  | 1.011 | 0.184  | Moderate   |
| BLIP (Fine-Tuned) | 0.357  | 0.287  | 1.227 | 0.220  | High       |
| BLIP-2 OPT 2.7B   | 0.375  | 0.275  | 1.251 | 0.217  | Highest    |
| BLIP-2 Flan-T5-XL | 0.330  | 0.272  | 1.118 | 0.216  | High       |

## 🔍 Key Insights
- Fine-tuning significantly improves performance across all metrics.
- BLIP-2 OPT 2.7B excels in CIDEr and CLIPScore, ideal for semantic quality.
- CLIPScore provides a unique perspective by evaluating vision-language alignment.

## 🛠️ Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Google Colab Pro+
- COCO 2017 Dataset
- pycocoevalcap
- OpenAI CLIP for CLIPScore

## 📄 Report
For methodology, analysis, and references, see: [`automated-image-captioning-report.pdf`](./automated-image-captioning-report.pdf)

## 🧾 References
- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
- [COCO Dataset](https://cocodataset.org)
- [CLIPScore](https://arxiv.org/abs/2104.08718)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---

> This project was submitted as part of a graduate-level course at George Mason University.
