# AWS Architecture Diagram

## Full System Architecture

```mermaid
graph LR
    subgraph Input
        SAT[(Satellite)]
        AIS[(AIS)]
    end
    
    subgraph Storage
        RAW[(S3 Raw)]
        PROC[(S3 Processed)]
    end
    
    subgraph Pipeline
        SF[Step Functions]
        L1[Ingest]
        L2[Detect]
        L3[Fuse]
        L4[Predict]
    end
    
    subgraph Output
        CW[CloudWatch]
        ATHENA[Athena]
    end
    
    SAT --> RAW
    AIS --> RAW
    RAW --> SF
    SF --> L1 --> L2 --> L3 --> L4
    L4 --> PROC
    PROC --> ATHENA
    SF --> CW
```

## Detailed Architecture

```mermaid
graph TD
    SAT[(Satellite Images)] --> RAW[(S3 Raw)]
    AIS[(AIS Data)] --> RAW
    
    RAW --> SF[Step Functions Pipeline]
    
    SF --> L1[Lambda: Ingest]
    L1 --> L2[Lambda: Detection]
    L2 --> L3[Lambda: Fuse]
    L3 --> L4[Lambda: Predict]
    L4 --> L5[Lambda: AIS]
    
    L2 --> PROC[(S3 Processed)]
    L3 --> PROC
    L4 --> PROC
    L5 --> PROC
    
    MODELS[(S3 Models)] --> L2
    MODELS --> L4
    
    PROC --> GLUE[Glue Catalog]
    GLUE --> ATHENA[Athena]
    
    SF --> CW[CloudWatch]
    L4 --> SNS[SNS Alerts]
    
    SM[SageMaker] --> MODELS
```

## Pipeline Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant SF as Step Functions
    participant L1 as IngestSatellite
    participant L2 as RunDetection
    participant L3 as FuseData
    participant L4 as RunForecasting
    participant L5 as ProcessAIS
    participant S3 as S3 Buckets
    participant SNS as SNS

    User->>SF: Start Execution
    SF->>L1: Invoke
    L1->>S3: Read satellite image
    L1-->>SF: Return metadata

    SF->>L2: Invoke
    L2->>L2: Run YOLO detection
    L2->>S3: Save detections
    L2-->>SF: Return detection_summary

    SF->>L3: Invoke with detection_summary
    L3->>S3: Read AIS data
    L3->>S3: Save fused features
    L3-->>SF: Return fused_features

    SF->>L4: Invoke with fused_features
    L4->>L4: Run forecasting model
    L4->>S3: Save predictions
    L4->>SNS: Send notification
    L4-->>SF: Return predictions

    SF->>L5: Invoke
    L5->>S3: Process AIS data
    L5-->>SF: Return AIS features

    SF-->>User: Pipeline Complete
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph Input["Input Data"]
        I1["Satellite Images<br/>2017-2024"]
        I2["AIS Ship Data<br/>Daily Records"]
        I3["News Sentiment<br/>Bloomberg"]
    end

    subgraph Processing["Processing Pipeline"]
        P1["Object Detection<br/>YOLO11"]
        P2["Feature Extraction<br/>Ship/Car Counts"]
        P3["Data Fusion<br/>Combine Sources"]
        P4["Forecasting<br/>ML Models"]
    end

    subgraph Output["Output"]
        O1["Trade Volume<br/>Prediction"]
        O2["Retail Index<br/>Prediction"]
        O3["Annotated Images<br/>Visualizations"]
    end

    I1 --> P1
    P1 --> P2
    I2 --> P3
    P2 --> P3
    P3 --> P4
    P4 --> O1
    P4 --> O2
    P1 --> O3
```

## AWS Services Overview

```mermaid
mindmap
    root((AWS Infrastructure))
        Storage
            S3 Raw Bucket
            S3 Processed Bucket
            S3 Models Bucket
        Compute
            Lambda Functions
                satellite-data-ingestion
                economic-forecast-detection
                economic-forecast-fuse
                economic-forecast-predict
                ais-data-ingestion
                news-sentiment-ingestion
            SageMaker
                Notebook Instance
                YOLO Endpoint
        Orchestration
            Step Functions Pipeline
            EventBridge Scheduler
        Analytics
            Glue Data Catalog
            Athena SQL Queries
        Monitoring
            CloudWatch Dashboard
            CloudWatch Alarms
            SNS Notifications
        Security
            IAM Roles
            VPC
```

## Component Details

### S3 Buckets

| Bucket | Purpose | Contents |
|--------|---------|----------|
| `economic-forecast-raw` | Raw data storage | Satellite images, AIS CSVs, News data |
| `economic-forecast-processed` | Processed outputs | Detections, Features, Predictions |
| `economic-forecast-models` | ML models | YOLO weights, Forecast models |

### Lambda Functions

| Function | Purpose | Trigger |
|----------|---------|---------|
| `satellite-data-ingestion` | Log satellite uploads | S3 / Step Functions |
| `economic-forecast-detection` | Run object detection | Step Functions |
| `economic-forecast-fuse` | Combine data sources | Step Functions |
| `economic-forecast-predict` | Generate forecasts | Step Functions |
| `ais-data-ingestion` | Process AIS data | Step Functions |
| `news-sentiment-ingestion` | Analyze news sentiment | S3 trigger |

### Step Functions Pipeline

```
Start
  │
  ▼
IngestSatellite ──► RunDetection ──► FuseData ──► RunForecasting ──► ProcessAIS
                                                                          │
                                                                          ▼
                                                                   PipelineComplete
```

## Cost Estimate (Monthly)

| Service | Usage | Estimated Cost |
|---------|-------|----------------|
| S3 | 10 GB storage | $0.23 |
| Lambda | 1000 invocations | $0.20 |
| Step Functions | 100 executions | $0.25 |
| SageMaker Notebook | 10 hours | $2.30 |
| CloudWatch | Logs & metrics | $1.00 |
| **Total** | | **~$4/month** |

---

*Generated for Real-Time Economic Forecasting Project*
*University of Maryland - DATA 650*
