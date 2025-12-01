#!/bin/bash
# ============================================================
# AWS Infrastructure Setup Script
# Economic Forecasting System
# ============================================================

set -e

# Configuration - UPDATE THESE
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PROJECT_NAME="economic-forecast"

echo "=============================================="
echo "🚀 Setting up AWS Infrastructure"
echo "   Region: $AWS_REGION"
echo "   Account: $AWS_ACCOUNT_ID"
echo "=============================================="

# ------------------------------------------------------------
# 1. Create SNS Topic
# ------------------------------------------------------------
echo ""
echo "📢 Creating SNS Topic..."

SNS_TOPIC_ARN=$(aws sns create-topic \
    --name ${PROJECT_NAME}-alerts \
    --region $AWS_REGION \
    --query 'TopicArn' --output text)

echo "   ✅ Created: $SNS_TOPIC_ARN"

# Subscribe email (update with your email)
# aws sns subscribe \
#     --topic-arn $SNS_TOPIC_ARN \
#     --protocol email \
#     --notification-endpoint your-email@example.com

# ------------------------------------------------------------
# 2. Create IAM Roles
# ------------------------------------------------------------
echo ""
echo "🔐 Creating IAM Roles..."

# Lambda Role
aws iam create-role \
    --role-name ${PROJECT_NAME}-lambda-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' 2>/dev/null || echo "   Role already exists"

# Attach policies to Lambda role
aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-lambda-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-lambda-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSNSFullAccess

aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-lambda-role \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-lambda-role \
    --policy-arn arn:aws:iam::aws:policy/AWSStepFunctionsFullAccess

LAMBDA_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${PROJECT_NAME}-lambda-role"
echo "   ✅ Lambda Role: $LAMBDA_ROLE_ARN"

# Step Functions Role
aws iam create-role \
    --role-name ${PROJECT_NAME}-stepfunctions-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "states.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' 2>/dev/null || echo "   Role already exists"

aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-stepfunctions-role \
    --policy-arn arn:aws:iam::aws:policy/AWSLambda_FullAccess

aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-stepfunctions-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSNSFullAccess

aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-stepfunctions-role \
    --policy-arn arn:aws:iam::aws:policy/AWSGlueConsoleFullAccess

STEPFUNCTIONS_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${PROJECT_NAME}-stepfunctions-role"
echo "   ✅ Step Functions Role: $STEPFUNCTIONS_ROLE_ARN"

# Wait for roles to propagate
echo "   ⏳ Waiting for roles to propagate..."
sleep 10

# ------------------------------------------------------------
# 3. Create Lambda Functions
# ------------------------------------------------------------
echo ""
echo "⚡ Creating Lambda Functions..."

# Package Lambda functions
cd lambda/s3_trigger
zip -r ../s3_trigger.zip lambda_function.py
cd ../..

cd lambda/process_ais
zip -r ../process_ais.zip lambda_function.py
cd ../..

cd lambda/forecast
zip -r ../forecast.zip lambda_function.py
cd ../..

# Create S3 Trigger Lambda
aws lambda create-function \
    --function-name ${PROJECT_NAME}-s3-trigger \
    --runtime python3.11 \
    --role $LAMBDA_ROLE_ARN \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda/s3_trigger.zip \
    --timeout 60 \
    --memory-size 256 \
    --environment "Variables={SNS_TOPIC_ARN=$SNS_TOPIC_ARN}" \
    --region $AWS_REGION 2>/dev/null || \
    aws lambda update-function-code \
        --function-name ${PROJECT_NAME}-s3-trigger \
        --zip-file fileb://lambda/s3_trigger.zip \
        --region $AWS_REGION

echo "   ✅ Created: ${PROJECT_NAME}-s3-trigger"

# Create AIS Processor Lambda
aws lambda create-function \
    --function-name ${PROJECT_NAME}-process-ais \
    --runtime python3.11 \
    --role $LAMBDA_ROLE_ARN \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda/process_ais.zip \
    --timeout 300 \
    --memory-size 512 \
    --region $AWS_REGION 2>/dev/null || \
    aws lambda update-function-code \
        --function-name ${PROJECT_NAME}-process-ais \
        --zip-file fileb://lambda/process_ais.zip \
        --region $AWS_REGION

echo "   ✅ Created: ${PROJECT_NAME}-process-ais"

# Create Forecast Lambda
aws lambda create-function \
    --function-name ${PROJECT_NAME}-predict \
    --runtime python3.11 \
    --role $LAMBDA_ROLE_ARN \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda/forecast.zip \
    --timeout 300 \
    --memory-size 512 \
    --region $AWS_REGION 2>/dev/null || \
    aws lambda update-function-code \
        --function-name ${PROJECT_NAME}-predict \
        --zip-file fileb://lambda/forecast.zip \
        --region $AWS_REGION

echo "   ✅ Created: ${PROJECT_NAME}-predict"

# ------------------------------------------------------------
# 4. Add S3 Trigger to Lambda
# ------------------------------------------------------------
echo ""
echo "🪣 Configuring S3 Trigger..."

# Add permission for S3 to invoke Lambda
aws lambda add-permission \
    --function-name ${PROJECT_NAME}-s3-trigger \
    --statement-id s3-trigger-permission \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::${PROJECT_NAME}-raw \
    --region $AWS_REGION 2>/dev/null || echo "   Permission already exists"

# Configure S3 bucket notification
aws s3api put-bucket-notification-configuration \
    --bucket ${PROJECT_NAME}-raw \
    --notification-configuration '{
        "LambdaFunctionConfigurations": [{
            "LambdaFunctionArn": "arn:aws:lambda:'$AWS_REGION':'$AWS_ACCOUNT_ID':function:'${PROJECT_NAME}'-s3-trigger",
            "Events": ["s3:ObjectCreated:*"],
            "Filter": {
                "Key": {
                    "FilterRules": [
                        {"Name": "prefix", "Value": "satellite/"},
                        {"Name": "suffix", "Value": ".jpg"}
                    ]
                }
            }
        }]
    }' 2>/dev/null || echo "   ⚠️ Configure S3 notification manually"

echo "   ✅ S3 trigger configured"

# ------------------------------------------------------------
# 5. Create Step Functions State Machine
# ------------------------------------------------------------
echo ""
echo "🔄 Creating Step Functions State Machine..."

# Replace placeholders in definition
sed -e "s/\${AWS_REGION}/$AWS_REGION/g" \
    -e "s/\${AWS_ACCOUNT_ID}/$AWS_ACCOUNT_ID/g" \
    -e "s/\${SNS_TOPIC_ARN}/$SNS_TOPIC_ARN/g" \
    step_functions/pipeline_definition.json > /tmp/pipeline_definition.json

aws stepfunctions create-state-machine \
    --name ${PROJECT_NAME}-pipeline \
    --definition file:///tmp/pipeline_definition.json \
    --role-arn $STEPFUNCTIONS_ROLE_ARN \
    --region $AWS_REGION 2>/dev/null || \
    aws stepfunctions update-state-machine \
        --state-machine-arn arn:aws:states:${AWS_REGION}:${AWS_ACCOUNT_ID}:stateMachine:${PROJECT_NAME}-pipeline \
        --definition file:///tmp/pipeline_definition.json \
        --region $AWS_REGION

STEPFUNCTION_ARN="arn:aws:states:${AWS_REGION}:${AWS_ACCOUNT_ID}:stateMachine:${PROJECT_NAME}-pipeline"
echo "   ✅ Created: $STEPFUNCTION_ARN"

# Update Lambda with Step Function ARN
aws lambda update-function-configuration \
    --function-name ${PROJECT_NAME}-s3-trigger \
    --environment "Variables={SNS_TOPIC_ARN=$SNS_TOPIC_ARN,STEP_FUNCTION_ARN=$STEPFUNCTION_ARN}" \
    --region $AWS_REGION

# ------------------------------------------------------------
# 6. Create EventBridge Rules
# ------------------------------------------------------------
echo ""
echo "⏰ Creating EventBridge Rules..."

# Daily AIS download rule
aws events put-rule \
    --name ${PROJECT_NAME}-daily-ais \
    --schedule-expression "cron(0 6 * * ? *)" \
    --state ENABLED \
    --region $AWS_REGION

aws events put-targets \
    --rule ${PROJECT_NAME}-daily-ais \
    --targets "Id"="1","Arn"="arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:function:${PROJECT_NAME}-process-ais" \
    --region $AWS_REGION

echo "   ✅ Created: ${PROJECT_NAME}-daily-ais"

# Weekly pipeline rule
aws events put-rule \
    --name ${PROJECT_NAME}-weekly-pipeline \
    --schedule-expression "cron(0 0 ? * SUN *)" \
    --state ENABLED \
    --region $AWS_REGION

echo "   ✅ Created: ${PROJECT_NAME}-weekly-pipeline"

# ------------------------------------------------------------
# 7. Create Glue Database and Crawler
# ------------------------------------------------------------
echo ""
echo "📊 Creating Glue Resources..."

# Create database
aws glue create-database \
    --database-input '{
        "Name": "economic_forecast_db",
        "Description": "Economic forecasting data catalog"
    }' \
    --region $AWS_REGION 2>/dev/null || echo "   Database already exists"

echo "   ✅ Created database: economic_forecast_db"

# Create Glue role
aws iam create-role \
    --role-name ${PROJECT_NAME}-glue-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "glue.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' 2>/dev/null || echo "   Role already exists"

aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-glue-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole

aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-glue-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

sleep 5

# Create crawler
aws glue create-crawler \
    --name ${PROJECT_NAME}-crawler \
    --role arn:aws:iam::${AWS_ACCOUNT_ID}:role/${PROJECT_NAME}-glue-role \
    --database-name economic_forecast_db \
    --targets '{
        "S3Targets": [
            {"Path": "s3://'${PROJECT_NAME}'-processed/features/"},
            {"Path": "s3://'${PROJECT_NAME}'-processed/predictions/"}
        ]
    }' \
    --region $AWS_REGION 2>/dev/null || echo "   Crawler already exists"

echo "   ✅ Created crawler: ${PROJECT_NAME}-crawler"

# ------------------------------------------------------------
# 8. Create CloudWatch Dashboard
# ------------------------------------------------------------
echo ""
echo "📈 Creating CloudWatch Dashboard..."

sed -e "s/\${AWS_REGION}/$AWS_REGION/g" \
    -e "s/\${AWS_ACCOUNT_ID}/$AWS_ACCOUNT_ID/g" \
    cloudwatch/dashboard.json > /tmp/dashboard.json

DASHBOARD_BODY=$(cat /tmp/dashboard.json | jq -c '.dashboardBody')

aws cloudwatch put-dashboard \
    --dashboard-name "Economic-Forecast-Monitor" \
    --dashboard-body "$DASHBOARD_BODY" \
    --region $AWS_REGION

echo "   ✅ Created dashboard: Economic-Forecast-Monitor"

# ------------------------------------------------------------
# 9. Create CloudWatch Alarms
# ------------------------------------------------------------
echo ""
echo "🚨 Creating CloudWatch Alarms..."

aws cloudwatch put-metric-alarm \
    --alarm-name "${PROJECT_NAME}-lambda-errors" \
    --alarm-description "Alert on Lambda errors" \
    --metric-name Errors \
    --namespace AWS/Lambda \
    --dimensions Name=FunctionName,Value=${PROJECT_NAME}-s3-trigger \
    --statistic Sum \
    --period 300 \
    --evaluation-periods 1 \
    --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --alarm-actions $SNS_TOPIC_ARN \
    --region $AWS_REGION

echo "   ✅ Created alarm: ${PROJECT_NAME}-lambda-errors"

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo ""
echo "=============================================="
echo "✅ AWS Infrastructure Setup Complete!"
echo "=============================================="
echo ""
echo "Resources Created:"
echo "  📢 SNS Topic: $SNS_TOPIC_ARN"
echo "  ⚡ Lambda Functions:"
echo "     - ${PROJECT_NAME}-s3-trigger"
echo "     - ${PROJECT_NAME}-process-ais"
echo "     - ${PROJECT_NAME}-predict"
echo "  🔄 Step Functions: $STEPFUNCTION_ARN"
echo "  ⏰ EventBridge Rules:"
echo "     - ${PROJECT_NAME}-daily-ais"
echo "     - ${PROJECT_NAME}-weekly-pipeline"
echo "  📊 Glue: economic_forecast_db + crawler"
echo "  📈 CloudWatch: Dashboard + Alarms"
echo ""
echo "Next Steps:"
echo "  1. Subscribe to SNS topic for email alerts"
echo "  2. Run Glue crawler to catalog existing data"
echo "  3. Test pipeline with: aws stepfunctions start-execution --state-machine-arn $STEPFUNCTION_ARN"
echo ""
