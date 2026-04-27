# Deployment Guide

## Docker Deployment

### Build Image
```bash
docker build -t active-learning:latest .
```

### Run Container
```bash
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  active-learning:latest
```

### Docker Compose
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## Kubernetes Deployment

### Deploy
```bash
kubectl apply -f k8s-deployment.yml
```

### Monitor
```bash
kubectl get deployment -n active-learning
kubectl logs -n active-learning -l app=al-system
kubectl top pod -n active-learning
```

### Scale
```bash
kubectl scale deployment al-system -n active-learning --replicas=3
```

### Cleanup
```bash
kubectl delete namespace active-learning
```

## Testing

### Unit Tests
```bash
pytest tests/test_data.py -v
```

### Integration Tests
```bash
pytest tests/test_integration.py -v
```

### All Tests
```bash
pytest tests/ -v --cov=.
```

## Production Checklist

- ✅ Docker image built and tested
- ✅ Kubernetes manifests validated
- ✅ Tests passing (unit & integration)
- ✅ GitHub Actions configured
- ✅ Documentation complete
- ✅ Results verified (87% accuracy)

## Performance

- CPU: Target 70% utilization
- Memory: Target 80% utilization
- Auto-scaling: 1-3 replicas based on load
