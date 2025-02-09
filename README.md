# **MHADFormer 🧠👨‍⚕️🤖**

**Author:** Francis Jesmar P. Montalbo  
**Affiliation:** Batangas State University 🎓  
**Email:** [francisjesmar.montalbo@g.batstate-u.edu.ph](mailto:francisjesmar.montalbo@g.batstate-u.edu.ph) | [francismontalbo@ieee.org](mailto:francismontalbo@ieee.org)

---

## **📌 Graphical Abstract**  
![Graphical Abstract](mhadformer_2025_final_graphical_abstract2.webp)

---

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

