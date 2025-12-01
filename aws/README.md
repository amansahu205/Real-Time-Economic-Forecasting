# 🏗️ AWS Infrastructure Setup

This directory contains all AWS service configurations for the Economic Forecasting System.

## 📊 Architecture Overview

```
                    ┌─────────────────┐
                    │   EventBridge   │
                    │  (Scheduler)    │
                    └────────┬────────┘
                             │ Daily/Weekly
                             ▼
┌──────────┐     ┌─────────────────────┐     ┌──────────┐
│    S3    │────▶│      Lambda         │────▶│   SNS    │
│ (Upload) │     │   (S3 Trigger)      │     │ (Alerts) │
└──────────┘     └─────────────────────┘     └──────────┘
                             │
                             ▼
                 ┌─────────────────────┐
                 │   Step Functions    │
                 │   (Orchestrator)    │
                 └─────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Lambda     │    │  SageMaker   │    │   Lambda     │
│ (AIS Process)│    │ (Detection)  │    │ (Forecast)   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                 ┌─────────────────────┐
                 │    Glue Catalog     │
                 │    + Athena Query   │
                 └─────────────────────┘
                             │
                             ▼
                 ┌─────────────────────┐
                 │    CloudWatch       │
                 │   (Monitoring)      │
                 └─────────────────────┘
```

## 📁 Directory Structure

```
aws/
├── lambda/                    # Lambda function code
│   ├── s3_trigger/           # Triggered on S3 uploads
│   ├── process_ais/          # AIS data processor
│   └── forecast/             # ML forecasting
│
├── step_functions/           # Pipeline orchestration
│   └── pipeline_definition.json
│
├── eventbridge/              # Scheduled jobs
│   └── rules.json
│
├── cloudwatch/               # Monitoring
│   ├── dashboard.json
│   └── alarms.json
│
├── glue/                     # Data catalog
│   └── crawler_config.json
│
├── iam/                      # IAM roles & policies
│   └── roles.json
│
├── setup_aws.sh              # Automated setup script
└── README.md
```

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
cd aws
chmod +x setup_aws.sh
./setup_aws.sh
```

### Option 2: Manual Setup via Console

Follow the step-by-step guide below.

---

## 📋 Manual Setup Guide

### 1️⃣ SNS - Notifications (5 min)

1. Go to **AWS Console → SNS → Topics → Create topic**
2. **Type**: Standard
3. **Name**: `economic-forecast-alerts`
4. Click **Create topic**
5. **Create subscription**:
   - Protocol: Email
   - Endpoint: your-email@example.com
6. Confirm email subscription

### 2️⃣ Lambda - S3 Trigger (10 min)

1. Go to **AWS Console → Lambda → Create function**
2. **Function name**: `economic-forecast-s3-trigger`
3. **Runtime**: Python 3.11
4. Click **Create function**
5. Copy code from `lambda/s3_trigger/lambda_function.py`
6. **Configuration → Environment variables**:
   - `SNS_TOPIC_ARN`: (your SNS topic ARN)
7. **Add trigger → S3**:
   - Bucket: `economic-forecast-raw`
   - Event type: All object create events

### 3️⃣ Lambda - AIS Processor (5 min)

1. Create function: `economic-forecast-process-ais`
2. Runtime: Python 3.11
3. Timeout: 5 minutes
4. Memory: 512 MB
5. Copy code from `lambda/process_ais/lambda_function.py`

### 4️⃣ Lambda - Forecaster (5 min)

1. Create function: `economic-forecast-predict`
2. Runtime: Python 3.11
3. Timeout: 5 minutes
4. Memory: 512 MB
5. Copy code from `lambda/forecast/lambda_function.py`

### 5️⃣ Step Functions - Pipeline (15 min)

1. Go to **AWS Console → Step Functions → Create state machine**
2. **Write your workflow in code**
3. Copy JSON from `step_functions/pipeline_definition.json`
4. Replace placeholders:
   - `${AWS_REGION}` → your region (e.g., `us-east-1`)
   - `${AWS_ACCOUNT_ID}` → your account ID
   - `${SNS_TOPIC_ARN}` → your SNS topic ARN
5. **Name**: `economic-forecasting-pipeline`
6. Create new IAM role

### 6️⃣ EventBridge - Scheduler (5 min)

1. Go to **AWS Console → EventBridge → Rules → Create rule**
2. **Rule 1: Daily AIS Download**
   - Name: `economic-forecast-daily-ais`
   - Schedule: `cron(0 6 * * ? *)`
   - Target: Lambda `economic-forecast-process-ais`

3. **Rule 2: Weekly Pipeline**
   - Name: `economic-forecast-weekly-pipeline`
   - Schedule: `cron(0 0 ? * SUN *)`
   - Target: Step Functions `economic-forecasting-pipeline`

### 7️⃣ Glue - Data Catalog (10 min)

1. Go to **AWS Console → Glue → Databases → Add database**
2. **Name**: `economic_forecast_db`

3. **Create Crawler**:
   - Name: `economic-forecast-crawler`
   - Data source: S3 path `s3://economic-forecast-processed/`
   - IAM role: Create new with S3 read access
   - Database: `economic_forecast_db`

4. **Run crawler** to catalog data

### 8️⃣ CloudWatch - Monitoring (10 min)

1. Go to **AWS Console → CloudWatch → Dashboards → Create dashboard**
2. **Name**: `Economic-Forecast-Monitor`
3. Add widgets from `cloudwatch/dashboard.json`

4. **Create Alarms**:
   - Lambda Errors alarm
   - Step Functions failure alarm

---

## 🧪 Testing

### Test S3 Trigger
```bash
# Upload a test file
aws s3 cp test_image.jpg s3://economic-forecast-raw/satellite/test/

# Check Lambda logs
aws logs tail /aws/lambda/economic-forecast-s3-trigger --follow
```

### Test Step Functions
```bash
# Start execution
aws stepfunctions start-execution \
    --state-machine-arn arn:aws:states:us-east-1:YOUR_ACCOUNT:stateMachine:economic-forecasting-pipeline \
    --input '{"action": "full_pipeline"}'
```

### Query with Athena
```sql
-- After running Glue crawler
SELECT * FROM economic_forecast_db.features LIMIT 10;
```

---

## 💰 Cost Estimates

| Service | Free Tier | Estimated Monthly Cost |
|---------|-----------|----------------------|
| Lambda | 1M requests | ~$0 (within free tier) |
| S3 | 5GB | ~$2-5 |
| Step Functions | 4000 transitions | ~$0 |
| SNS | 1M publishes | ~$0 |
| Glue | 1M objects | ~$1 |
| CloudWatch | Basic | ~$0 |
| **Total** | | **~$3-10/month** |

---

## 🔧 Troubleshooting

### Lambda Timeout
- Increase timeout in Configuration
- Check if S3 files are too large

### Step Functions Failed
- Check CloudWatch logs for each step
- Verify IAM permissions

### Glue Crawler Empty
- Verify S3 path has data
- Check IAM role has S3 access

---

## 📚 AWS Services Used

| Service | Purpose |
|---------|---------|
| **S3** | Data storage (raw, processed, models) |
| **Lambda** | Serverless compute for processing |
| **Step Functions** | Pipeline orchestration |
| **EventBridge** | Scheduled triggers |
| **SNS** | Notifications and alerts |
| **Glue** | Data catalog for Athena |
| **Athena** | SQL queries on S3 data |
| **CloudWatch** | Monitoring and logging |
| **IAM** | Security and access control |
| **SageMaker** | ML model training and inference |

---

**Total Services: 10 AWS Services** ✅
