# Pushback Sensitivity in Language Models

## Hugging Face Token

To use the model in this project, you will need a Hugging Face account and a User Access Token. This token serves two primary purposes:

1. **Model Access:** The model we are using is a "gated model." This means you must explicitly request access and agree to the creator's terms of use before you can download it. Your token authenticates your account and verifies that you have been granted permission.
2. **Download Speeds & Rate Limits:** Passing an authenticated token prevents you from hitting anonymous download rate limits, which can severely throttle your connection. 

### How to get your token:

1. **Create an Account:** If you don't have one, register at [Hugging Face](https://huggingface.co/join).
2. **Request Model Access:** Navigate to the model's page on Hugging Face *(https://huggingface.co/meta-llama/Llama-3.2-3B)* and click the button to agree to the terms and request access. 
3. **Generate a Token:** Go to your [Hugging Face Access Tokens page](https://huggingface.co/settings/tokens). Click **"New token"**, give it a name, and generate a token with `Read` permissions.

Once you have your token copied, see the [Setup Environment Variables](#setup-environment-variables) section below to configure it in your local environment.

## Setup Environment Variables

Before running the project, you need to configure your local environment variables to ensure everything runs smoothly. Follow the below instructions:

1. Copy the provided `.env.sample` file and rename it to `.env`
2. Fill in all required fields in this new `.env` file
