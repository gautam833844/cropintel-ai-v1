# Vercel Deployment Guide

## Prerequisites
1. **Vercel Account**: Sign up at https://vercel.com (free)
2. **GitHub Account**: Push your code to GitHub
3. **Model File**: Ensure `crop_recommendation_model.pkl` is in your project root

## Deployment Steps

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit for Vercel deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Connect to Vercel
1. Go to https://vercel.com/new
2. Click "Import Git Repository"
3. Paste your GitHub repo URL
4. Click "Import"

### Step 3: Configure Project Settings
- **Framework Preset**: Other (Python)
- **Root Directory**: / (or leave as is)
- **Environment Variables**: Add any required env vars
- Click "Deploy"

### Step 4: Your app will be live!
Vercel will provide you with a URL like: `https://your-project.vercel.app`

## Important Notes

⚠️ **Model File Size**: Make sure your model file is under Vercel's serverless function size limits:
- Standard: 250MB uncompressed
- Pro plan: 3GB uncompressed

📦 **Large Files**: If your model is large, consider:
1. Using cloud storage (AWS S3, Google Cloud Storage)
2. Using Vercel's blob storage (add to environment)
3. Compressing the model with gzip

🔒 **Environment Variables**: For production:
- Add API keys and secrets in Vercel dashboard
- Set `SESSION_COOKIE_SECURE = True` in production
- Configure proper CORS if needed

## Troubleshooting

**Build Fails**: Check build logs in Vercel dashboard
**Model Not Found**: Ensure model file is in root directory
**Slow Initial Load**: Cold starts on serverless are normal (1-5 sec)
**Large Deployments**: Use Vercel's `functions.maxDuration` setting

## Monitoring
- View logs: Dashboard → Project → Deployments
- Monitor errors: Integrations → Error tracking
- Check performance: Analytics in dashboard

## Need More Help?
- Vercel Docs: https://vercel.com/docs
- Python on Vercel: https://vercel.com/docs/functions/python
