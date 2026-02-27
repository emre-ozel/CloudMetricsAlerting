# Makefile — build and deploy the two-Lambda alerting pipeline with AWS SAM.
#
# Prerequisites:
#   pip install aws-sam-cli
#   aws configure   (or set AWS_PROFILE)
#
# Usage:
#   make layer      — build the shared Python deps layer
#   make build      — sam build (packages both Lambdas + layer)
#   make deploy     — sam deploy (first run creates guided wizard)
#   make invoke-retrain   — manually invoke the retrain Lambda
#   make invoke-inference — manually invoke the inference Lambda
#   make logs-retrain     — tail CloudWatch logs for retrain
#   make logs-inference   — tail CloudWatch logs for inference
#   make destroy    — delete the entire CloudFormation stack

STACK_NAME   ?= cloud-metrics-alerting
REGION       ?= us-east-1
ENV          ?= prod
ALERT_EMAIL  ?= alerts@example.com
PROFILE      ?= default

SAMFLAGS = --stack-name $(STACK_NAME) \
           --region $(REGION)         \
           --profile $(PROFILE)

# ── Layer build ───────────────────────────────────────────────────────────────
.PHONY: layer
layer:
	@echo "Building shared Lambda layer…"
	rm -rf layer/python
	mkdir -p layer/python
	pip install                                     \
	    --platform manylinux2014_aarch64            \
	    --implementation cp                         \
	    --python-version 3.12                       \
	    --only-binary=:all:                         \
	    --target layer/python                       \
	    numpy scikit-learn lightgbm xgboost joblib boto3
	# Copy shared utilities into the layer
	cp -r shared layer/python/shared
	@echo "Layer ready at layer/python/"

# ── SAM build ─────────────────────────────────────────────────────────────────
.PHONY: build
build: layer
	sam build --template lambda/template.yaml --use-container

# ── SAM deploy ────────────────────────────────────────────────────────────────
.PHONY: deploy
deploy: build
	sam deploy $(SAMFLAGS)                          \
	    --capabilities CAPABILITY_NAMED_IAM         \
	    --parameter-overrides                        \
	        Environment=$(ENV)                       \
	        AlertEmail=$(ALERT_EMAIL)                \
	    --confirm-changeset

# ── Manual invocations ────────────────────────────────────────────────────────
.PHONY: invoke-retrain
invoke-retrain:
	aws lambda invoke                                       \
	    --function-name cloud-metrics-retrain-$(ENV)        \
	    --payload '{"trigger":"manual"}'                    \
	    --cli-binary-format raw-in-base64-out               \
	    --region $(REGION) --profile $(PROFILE)             \
	    /tmp/retrain_response.json
	cat /tmp/retrain_response.json | python3 -m json.tool

.PHONY: invoke-inference
invoke-inference:
	aws lambda invoke                                       \
	    --function-name cloud-metrics-inference-$(ENV)      \
	    --payload '{"trigger":"manual"}'                    \
	    --cli-binary-format raw-in-base64-out               \
	    --region $(REGION) --profile $(PROFILE)             \
	    /tmp/inference_response.json
	cat /tmp/inference_response.json | python3 -m json.tool

# ── Log tailing ───────────────────────────────────────────────────────────────
.PHONY: logs-retrain
logs-retrain:
	sam logs --name RetrainFunction $(SAMFLAGS) --tail

.PHONY: logs-inference
logs-inference:
	sam logs --name InferenceFunction $(SAMFLAGS) --tail

# ── Destroy ───────────────────────────────────────────────────────────────────
.PHONY: destroy
destroy:
	aws cloudformation delete-stack     \
	    --stack-name $(STACK_NAME)      \
	    --region $(REGION)              \
	    --profile $(PROFILE)
	@echo "Stack deletion initiated. S3 bucket is retained (DeletionPolicy: Retain)."
