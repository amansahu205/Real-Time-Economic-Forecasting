# 🎤 Presentation Scripts for Demo Notebooks

**Total Time: ~20-25 minutes**

---

## 📋 Overview

| Demo | Topic | Duration | Presenter |
|------|-------|----------|-----------|
| Demo 1 | YOLO Training | 3-4 min | |
| Demo 2 | Object Detection | 5-6 min | |
| Demo 3 | AIS Maritime Data | 3-4 min | |
| Demo 4 | Data Fusion | 3-4 min | |
| Demo 5 | Economic Forecasting | 5-6 min | |

---

# 🎯 Demo 1: YOLO Model Training

## Script (~3-4 minutes)

### Opening (30 sec)
> "Let's start with how we trained our AI models to detect ships and vehicles in satellite imagery. We're using YOLO - You Only Look Once - which is a state-of-the-art real-time object detection model."

### Cell 1: Setup (run while talking)
> "First, we set up our environment. The notebook automatically detects if we're running in AWS SageMaker and installs the required dependencies."

**Expected Output:**
```
📦 Installing dependencies...
✅ Dependencies installed
🌩️  Running in AWS SageMaker
✅ Setup complete | S3: True | YOLO: True
```

### Cell 2: YOLO Architecture (30 sec)
> "YOLO works by dividing an image into a grid and predicting bounding boxes and class probabilities simultaneously. This makes it extremely fast - hence 'You Only Look Once'. We're using YOLO11, the latest version with about 2.6 million parameters."

**Key Points to Mention:**
- Pre-trained on COCO dataset (80 classes)
- We fine-tune for satellite-specific objects
- Nano version for demo speed

### Cell 3: Training Datasets (45 sec)
> "We trained on two specialized datasets:
> 1. **DOTA** - A large-scale dataset for object detection in aerial images. It contains ships, harbors, storage tanks, and vehicles.
> 2. **xView** - Overhead imagery with similar classes.
> 
> The labels are in YOLO format - each object has a class ID and normalized bounding box coordinates."

### Cell 4: Training Demo (1 min)
> "For this demo, we'll run just 5 epochs to show the training process. In production, we trained for 100+ epochs to achieve our best accuracy."

**While training runs, explain:**
- Loss decreasing = model learning
- mAP (mean Average Precision) = detection accuracy
- Training on GPU accelerates this significantly

### Closing (30 sec)
> "After full training, our port detection model achieved about 85% mAP, and our retail model achieved about 80% mAP. These trained models are stored in S3 and will be used in the next demo for actual detection."

---

# 🔍 Demo 2: Object Detection

## Script (~5-6 minutes)

### Opening (30 sec)
> "Now let's use our trained models to actually detect ships and vehicles in real satellite imagery. We'll process images from the Port of LA and Mall of America."

### Cell 1: Setup (run while talking)
> "Setting up and loading our trained models from S3..."

### Cell 2: TiledDetector Explanation (1 min)
> "Here's a key technical challenge: our satellite images are very large - around 8000x5000 pixels. But YOLO processes images at 640x640. If we just resize, small objects like ships become tiny and get missed.
>
> Our solution is **tiled detection**:
> 1. Split the image into overlapping 1024x1024 tiles
> 2. Run detection on each tile
> 3. Merge results and remove duplicates using Non-Maximum Suppression
>
> This preserves detail for small object detection."

### Cell 3: Load Models (30 sec)
> "We're downloading our trained models from S3. The port model was trained on the DOTA dataset for ship detection, and the retail model was trained for car detection in parking lots."

**Expected Output:**
```
📥 Downloading port model from S3...
✅ Downloaded to /tmp/port_best.pt
📥 Downloading retail model from S3...
✅ Downloaded to /tmp/retail_best.pt
✅ Port detector ready (DOTA-trained for ships)
✅ Retail detector ready (trained for cars)
```

### Cell 4-5: Port of LA Detection (1.5 min)
> "Let's detect ships at the Port of LA. This is a 2024 satellite image from Google Earth."

**While processing, explain:**
> "You can see the image is being split into 60 tiles. Each tile is processed independently, then results are merged."

**After results appear:**
> "We detected 97 objects across the port area. The model identifies:
> - **Ships** (green boxes) - cargo vessels in the harbor
> - **Large vehicles** - trucks and equipment
> - **Storage tanks** - fuel storage facilities
>
> Notice the 2020 data shows significantly more ships - this was the COVID supply chain backup when ships were waiting weeks to unload."

### Cell 6-7: Mall of America Detection (1.5 min)
> "Now let's look at retail activity. This is the Mall of America parking lot."

**After results:**
> "We detected vehicles in the parking lot. By counting cars across different years, we can track retail activity trends.
>
> Key insight: In 2020, car counts dropped by about 38% - directly reflecting the COVID lockdown impact on retail."

### Cell 8: Historical Trends (1 min)
> "Here's the summary across all years. The bar charts show:
> - **Port of LA**: Ship counts spiked in 2020-2021 (supply chain crisis)
> - **Mall of America**: Car counts dropped sharply in 2020 (lockdowns)
>
> These satellite-derived metrics give us leading indicators of economic activity - often weeks before official statistics are released."

---

# 🚢 Demo 3: AIS Maritime Data

## Script (~3-4 minutes)

### Opening (30 sec)
> "Satellite images give us snapshots, but AIS - Automatic Identification System - gives us continuous ship tracking. Every commercial vessel broadcasts its position, speed, and type every few seconds."

### Cell 1: Setup (run quickly)

### Cell 2: AIS Explanation (45 sec)
> "AIS data includes:
> - **MMSI**: Unique ship identifier
> - **Position**: Latitude/longitude
> - **Speed and heading**: Direction of travel
> - **Vessel type**: Cargo (70-79), Tanker (80-89), Passenger (60-69)
>
> Cargo ships and tankers are our key economic indicators - they carry goods."

### Cell 3: Port of LA Bounding Box (30 sec)
> "We filter AIS data to the Port of LA region - a geographic bounding box around the port. This captures all ships entering, leaving, or waiting at the port."

**Show the map visualization**

### Cell 4-5: Load and Display AIS Data (1 min)
> "Here's a sample of AIS records. Each row is a ship position broadcast. We aggregate these into daily and yearly metrics:
> - Total unique ships
> - Cargo vs tanker breakdown
> - Average dwell time (how long ships wait)"

### Cell 6: Ship Traffic Analysis (1 min)
> "The visualization shows:
> - **Vessel type distribution**: ~60% cargo, ~20% tankers
> - **Traffic patterns**: Daily arrivals and departures
> - **Dwell time spike in 2020**: Ships waiting 3.8 days vs normal 2.2 days
>
> This dwell time metric was a leading indicator of the supply chain crisis - we could see the backup building before it made headlines."

---

# 🔗 Demo 4: Data Fusion

## Script (~3-4 minutes)

### Opening (30 sec)
> "Now we combine satellite detection with AIS tracking. Each source has strengths and weaknesses - together they're more powerful."

### Cell 1: Setup (run quickly)

### Cell 2: Why Fusion? (1 min)
> "Look at this comparison:
>
> **Satellite alone:**
> - ✅ Visual proof of ships
> - ❌ Just a snapshot in time
> - ❌ Can't identify vessel type
>
> **AIS alone:**
> - ✅ Continuous tracking
> - ✅ Vessel type and cargo info
> - ❌ Can be spoofed or turned off
>
> **Fused data:**
> - ✅ Validated ship counts
> - ✅ Vessel type breakdown
> - ✅ Dwell time analysis
> - ✅ Complete picture for forecasting"

### Cell 3-4: Load Both Sources (30 sec)
> "We load satellite detection results and AIS tracking data, both aggregated by year."

### Cell 5: Merge Data (1 min)
> "We merge on year and calculate derived features:
> - Ships per image (normalized satellite count)
> - Cargo ratio (% of ships that are cargo vessels)
> - Combined activity index
>
> The correlation between satellite ship counts and AIS counts is about 0.85 - they validate each other."

### Cell 6: Validation Visualization (1 min)
> "This scatter plot shows satellite vs AIS ship counts. The strong correlation confirms our detection is accurate.
>
> The 2020 outlier shows both sources captured the supply chain surge - this wasn't noise, it was real economic signal."

---

# 🔮 Demo 5: Economic Forecasting

## Script (~5-6 minutes)

### Opening (30 sec)
> "Finally, we use our satellite and AIS features to predict actual economic indicators - trade volume and retail activity."

### Cell 1: Setup (run while talking)
> "Loading scikit-learn for our ML models..."

### Cell 2: Load Data (30 sec)
> "We load our detection results and merge with ground truth economic data:
> - **Trade volume**: Port of LA throughput in million TEUs (from official port statistics)
> - **Foot traffic**: Retail activity index (from mobility data)"

### Cell 3: Feature Engineering (1 min)
> "We create features from our satellite detections:
>
> **For trade forecasting:**
> - Total ships detected
> - Ships per image (normalized)
> - Year-over-year growth
> - 2-year moving average
>
> **For retail forecasting:**
> - Cars per image
> - Growth rate
> - Moving average
>
> These derived features help capture trends and momentum."

### Cell 4: Train Trade Model (1 min)
> "We train four different models and compare:
> - Linear Regression
> - Ridge Regression
> - Random Forest
> - Gradient Boosting
>
> Training on 2017-2022, testing on 2023-2024."

**After results:**
> "Random Forest performs best with MAE of about 0.15 million TEUs - that's less than 2% error on trade volume predictions."

### Cell 5: Train Retail Model (30 sec)
> "Same approach for retail foot traffic..."

**After results:**
> "Again, Random Forest wins with about 2 index points error."

### Cell 6: Visualization (1.5 min)
> "These charts tell the story:
>
> **Top left**: Trade volume - actual vs predicted. Our model captures the 2020 dip and 2021 surge.
>
> **Top right**: Ship detections - our input feature. Notice how it correlates with trade volume.
>
> **Bottom left**: Retail foot traffic - the 2020 crash is clearly visible.
>
> **Bottom right**: Cars per image - directly tracks retail activity.
>
> The key insight: satellite-derived features are **leading indicators**. We can see economic changes 2-4 weeks before official statistics are released."

### Cell 7: Prediction Results (30 sec)
> "For 2023-2024 predictions:
> - Trade volume: ~3% error
> - Foot traffic: ~2% error
>
> This is remarkably accurate given we're using satellite imagery to predict economic activity."

### Cell 8: Feature Importance (30 sec)
> "Feature importance shows 'ships per image' and 'cars per image' are the most predictive features - exactly what we'd expect. The normalized counts matter more than raw totals."

---

# 🎯 Closing Summary

## Key Takeaways (1 min)

> "To summarize what we've built:
>
> 1. **AI-powered detection**: YOLO models trained on satellite imagery detect ships and vehicles with 80-85% accuracy
>
> 2. **Multi-source fusion**: Combining satellite snapshots with AIS tracking gives us validated, comprehensive data
>
> 3. **Economic forecasting**: Our models predict trade volume and retail activity with less than 5% error
>
> 4. **Leading indicators**: We can see economic signals 2-4 weeks before official statistics
>
> 5. **Cloud-native**: Everything runs on AWS - S3 for data, SageMaker for ML, scalable and production-ready
>
> **Business value**: Investment firms, supply chain managers, and policy makers can use this for earlier, data-driven decisions."

---

# 📊 Expected Results Summary

## Demo 2: Detection Results

| Location | Objects Detected | Key Finding |
|----------|------------------|-------------|
| Port of LA (2024) | ~97 objects | Ships, vehicles, storage tanks |
| Port of LA (2020) | ~222 ships | +88% vs 2019 (COVID surge) |
| Mall of America (2020) | ~485 cars | -38% vs 2019 (lockdown) |

## Demo 5: Forecasting Results

| Model | Trade MAE | Retail MAE |
|-------|-----------|------------|
| Linear Regression | ~0.25 | ~4.5 |
| Ridge Regression | ~0.22 | ~4.2 |
| **Random Forest** | **~0.15** | **~2.1** |
| Gradient Boosting | ~0.18 | ~2.8 |

---

# 🛠️ Troubleshooting During Demo

## If a cell fails:

1. **NameError**: Run the setup cell first
2. **S3 download fails**: Check internet connection
3. **Model not found**: Verify S3 bucket permissions
4. **Slow processing**: Normal for large images (60 tiles)

## Backup talking points if something breaks:

> "While this is processing, let me explain the architecture..."
> "The key insight here is..."
> "In production, this runs automatically via AWS Step Functions..."

---

# 📁 Files Reference

```
notebooks/demo/
├── Demo_1_YOLO_Training.ipynb      # Model training
├── Demo_2_Object_Detection.ipynb   # Ship/car detection
├── Demo_3_AIS_Data.ipynb           # Maritime tracking
├── Demo_4_Data_Fusion.ipynb        # Combine sources
├── Demo_5_Forecasting.ipynb        # ML predictions
└── README.md
```

---

**Good luck with your presentation! 🚀**
