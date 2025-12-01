# Preprocessing Scripts (Archive)

**Status:** ✅ All preprocessing complete - these scripts are no longer needed for the current workflow.

---

## 📋 What's Here:

These scripts were used **once** to prepare data before model training. They are kept for reference and documentation purposes.

### **1. create_optimized_datasets.py**
```
Purpose: Filter and remap xView dataset to create focused datasets
Created:
  - retail_2class/ (car, equipment)
  - city_3class/ (car, truck, warehouse)
  - industrial_2class/ (warehouse, equipment)

Status: ✅ Complete - Datasets created and used for training
Output: data/raw/satellite/xview-economic-simple/
```

### **2. create_retail_2class.py**
```
Purpose: Create retail 2-class dataset specifically
Created:
  - retail_2class.yaml
  - Filtered images with only car + equipment

Status: ✅ Complete - Dataset used for retail model training
Output: data/raw/satellite/xview-economic-simple/retail_2class/
```

### **3. download_planetary_computer_imagery.py**
```
Purpose: Download satellite imagery from Microsoft Planetary Computer
Downloaded:
  - NAIP (1m resolution, US locations): 10.7 GB
  - Sentinel-2 L2A (10m resolution, global): 103 GB
  - 50 locations × 8 years (2017-2024)

Status: ✅ Complete - 145 GB imagery downloaded
Output: data/raw/satellite/naip/ and sentinel-2-l2a/
```

### **4. preprocess_dota_ports.py**
```
Purpose: Preprocess DOTA dataset for port activity detection
Created:
  - Converted oriented bounding boxes to axis-aligned
  - Tiled large images to 1024×1024
  - Filtered 5 port-relevant classes
  - Generated ports_dota.yaml

Status: ✅ Complete - Dataset used for excellent ports model (72% mAP!)
Output: data/raw/satellite/dota/ports/
```

---

## ❓ When Would You Need These Again?

### **Re-run download_planetary_computer_imagery.py if:**
- You need imagery for new locations
- You want more recent data (2025+)
- You need different time periods

### **Re-run dataset creation scripts if:**
- You want to try different class combinations
- You need to filter/remap for new activities
- You're experimenting with class balancing

### **Otherwise:**
**These scripts are done! Focus on inference and forecasting.**

---

## 🎯 Current Workflow:

```
✅ Data preprocessed (DONE - these scripts)
    ↓
✅ Models trained (DONE - Activity_Training_Complete.ipynb)
    ↓
⭐ Run inference (NEXT - Inference_Satellite_Activity.ipynb)
    ↓
📊 Analyze results (Analyze_Activity_Metrics.ipynb)
    ↓
🔮 Forecasting (Future)
```

---

**Keep these for reference, but you won't need to run them again for the current project!**
