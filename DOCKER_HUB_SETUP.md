# Docker Hub CI/CD Setup

This document explains how to configure and use the GitHub Action for automatically building and pushing Docker images to Docker Hub.

## Prerequisites

You need to configure the following secrets in your GitHub repository:

### Required GitHub Secrets

1. **DOCKER_HUB_USERNAME**: Your Docker Hub username
2. **DOCKER_HUB_TOKEN**: Your Docker Hub access token (NOT your password)

### How to Create a Docker Hub Access Token

1. Log in to [Docker Hub](https://hub.docker.com/)
2. Click on your username in the top right corner
3. Select **Account Settings**
4. Go to **Security** tab
5. Click **New Access Token**
6. Give it a descriptive name (e.g., "GitHub Actions")
7. Set permissions (Read, Write, Delete recommended for CI/CD)
8. Click **Generate**
9. **Copy the token immediately** (you won't be able to see it again)

### How to Add Secrets to GitHub Repository

1. Go to your GitHub repository
2. Click on **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add each secret:
   - Name: `DOCKER_HUB_USERNAME`, Value: your Docker Hub username
   - Name: `DOCKER_HUB_TOKEN`, Value: the access token you generated

## Workflow Triggers

The GitHub Action will automatically run when:

1. **Push to main branch**: Builds and pushes with `latest` tag and branch name
2. **Push a version tag** (e.g., `v1.0.0`): Builds and pushes with semantic version tags
3. **Manual trigger**: Go to Actions → Docker Build and Push → Run workflow

## Docker Image Tags

The workflow automatically creates multiple tags for easy access:

- `latest` - Always points to the most recent build from main branch
- `main` - Current state of the main branch
- `v1.0.0`, `v1.0`, `v1` - Semantic version tags (when you push version tags)
- `main-<sha>` - Branch name with git commit SHA for traceability

## Usage

### Automatic Build (Push to Main)

Simply push your code to the main branch:

```bash
git push origin main
```

The action will automatically build and push the Docker image.

### Tagged Release

To create a versioned release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This will create Docker images with tags: `v1.0.0`, `v1.0`, `v1`, and `latest`.

### Manual Build

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. Select **Docker Build and Push to Docker Hub** workflow
4. Click **Run workflow**
5. Optionally enter a custom tag name
6. Click **Run workflow** button

## Pulling the Image

Once the workflow completes, you can pull the image from Docker Hub:

```bash
# Pull the latest version
docker pull <your-dockerhub-username>/ml-depth-pro:latest

# Pull a specific version
docker pull <your-dockerhub-username>/ml-depth-pro:v1.0.0

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  <your-dockerhub-username>/ml-depth-pro:latest
```

## Viewing Build Logs

1. Go to the **Actions** tab in your GitHub repository
2. Click on the workflow run you want to inspect
3. View logs for each step of the build process

## Troubleshooting

### Authentication Failed

- Verify your `DOCKER_HUB_USERNAME` and `DOCKER_HUB_TOKEN` secrets are correctly set
- Ensure the access token has not expired
- Regenerate the token if necessary

### Build Failed

- Check the build logs in the Actions tab
- Ensure the Dockerfile is valid
- Verify all necessary files are included (check .dockerignore)

### Image Not Found on Docker Hub

- Verify the repository name matches your Docker Hub username
- Check if you have permission to push to that repository
- Ensure the repository exists on Docker Hub (it will be created automatically on first push if you have permissions)

## Advanced Configuration

### Custom Repository Name

If you want to push to a different repository name, update the `DOCKER_HUB_REPO` environment variable in `.github/workflows/docker-publish.yml`:

```yaml
env:
  DOCKER_HUB_REPO: your-username/your-custom-repo-name
```

### Multi-Platform Builds

To build for multiple architectures (e.g., ARM), modify the `platforms` in the workflow:

```yaml
platforms: linux/amd64,linux/arm64
```

Note: Multi-platform builds take longer to complete.

## Security Notes

- Never commit Docker Hub credentials directly in the code
- Use GitHub Secrets for all sensitive information
- Regularly rotate your Docker Hub access tokens
- Use access tokens with minimal required permissions
- Consider using Docker Hub's automated builds as an alternative approach
