# Imgour Object Removal - Replicate Deployment

This directory contains everything you need to deploy your object removal model to Replicate.

## What You've Got

- **cog.yaml**: Configuration file that tells Replicate what dependencies to install
- **predict.py**: The prediction interface that wraps your LaMa inpainting model
- This uses the same LaMa model that's currently running in your `ml-inpaint-service`

## Prerequisites

1. **Install Docker** (required for Cog)
   - Mac: Download from https://www.docker.com/products/docker-desktop
   - Make sure Docker Desktop is running

2. **Install Cog** (Replicate's packaging tool)
   ```bash
   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
   sudo chmod +x /usr/local/bin/cog
   ```

3. **Create a Replicate account**
   - Go to https://replicate.com/
   - Sign up (it's free to start)

## Step-by-Step Deployment

### 1. Test Locally (Optional but Recommended)

First, navigate to this directory:
```bash
cd replicate-deployment
```

Build the Cog container (this will take a while the first time - it's downloading models):
```bash
cog build
```

Test the prediction locally:
```bash
cog predict -i image=@/path/to/your/test-image.jpg -i mask=@/path/to/your/mask.png
```

### 2. Create Your Model on Replicate

1. Go to https://replicate.com/create
2. Choose a name for your model (e.g., `yourname/object-removal`)
3. Add a description: "AI-powered object removal using LaMa inpainting"
4. Set visibility (Public or Private)

### 3. Push to Replicate

First, log in to Replicate:
```bash
cog login
```

Then push your model:
```bash
cog push r8.im/yourname/object-removal
```

Replace `yourname/object-removal` with your actual Replicate username and model name.

This will:
- Build your Docker container
- Upload it to Replicate
- Make your model available via API

### 4. Use Your Model

Once deployed, you can use it via:

**Web UI**: https://replicate.com/yourname/object-removal

**API** (Node.js):
```javascript
import Replicate from "replicate";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

const output = await replicate.run(
  "yourname/object-removal",
  {
    input: {
      image: "https://example.com/image.jpg",
      mask: "https://example.com/mask.png"
    }
  }
);
```

**API** (Python):
```python
import replicate

output = replicate.run(
  "yourname/object-removal",
  input={
    "image": "https://example.com/image.jpg",
    "mask": "https://example.com/mask.png"
  }
)
```

**API** (HTTP):
```bash
curl -X POST https://api.replicate.com/v1/predictions \\
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "version": "YOUR_MODEL_VERSION_ID",
    "input": {
      "image": "https://example.com/image.jpg",
      "mask": "https://example.com/mask.png"
    }
  }'
```

## Pricing

Replicate charges based on GPU time used:
- You only pay when people use your model
- First deploys are free for testing
- Production usage: ~$0.000725/second on Nvidia T4 GPU
- Check current pricing: https://replicate.com/pricing

## Updating Your Model

To push updates:
1. Make changes to `predict.py` or `cog.yaml`
2. Run `cog push r8.im/yourname/object-removal` again
3. Replicate will create a new version

## Troubleshooting

**"Docker is not running"**
- Open Docker Desktop and make sure it's running

**"cog: command not found"**
- Make sure you installed Cog correctly
- Try: `which cog` to verify installation

**Build fails with CUDA errors**
- This is normal on Mac (no NVIDIA GPU)
- The build still works - it will use CPU locally
- On Replicate's servers, it will use GPU

**Model takes too long**
- Adjust `hd_strategy_resize_limit` parameter (lower = faster, but lower quality)
- Default is 800px which is a good balance

## What's Next?

Once your model is on Replicate:
1. Share the URL with users
2. Integrate it into your Next.js app via API
3. Let Replicate handle all the scaling, GPU management, and infrastructure
4. You can optionally replace your current `ml-inpaint-service` endpoint with the Replicate API

## Support

- Replicate Docs: https://replicate.com/docs
- Cog GitHub: https://github.com/replicate/cog
- Discord: https://discord.gg/replicate
