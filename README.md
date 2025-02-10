# **MHADFormer 🧠👨‍⚕️🤖**

**Author:** Francis Jesmar P. Montalbo  
**Affiliation:** Batangas State University 🎓  
**Email:** [francisjesmar.montalbo@g.batstate-u.edu.ph](mailto:francisjesmar.montalbo@g.batstate-u.edu.ph) | [francismontalbo@ieee.org](mailto:francismontalbo@ieee.org)

---

## **📌 Graphical Abstract**  
![Graphical Abstract](mhadformer_2025_graphical_abstract.webp)

---

## 📢 Full Details Coming Soon  
The complete details, including model architecture, training methodology, and benchmark comparisons, will be made available **upon formal publication**. Stay tuned for updates!  

For any inquiries or collaborations, feel free to reach out via email.  

---

📌 **License:** To be determined upon publication  
📌 **Citation:** Citation details will be provided once the paper is published.  



## **📄 Abstract**  
**  

---

## **🔑 Keywords**  
*(List relevant keywords for indexing and searchability.)*  
- 🧠  
- 🔄  
- 🏥 
- 🖼️ 
- 🤖   

---

## Quick Tutorial: Using the MHADFormer Model

The MHADFormer model is implemented as a class for modularity and ease of use. Below is a short example demonstrating how to import, instantiate, and use the model in your project.

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/mhadformer_release_2025.git
cd mhadformer_release_2025
pip install -r requirements.txt
```

---

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from utils.blocks import *

# Instantiate the model
model = MHADFormer(num_classes=5, image_size=224)
model.build(input_shape=(None, 224, 224, 3))
model.summary()

