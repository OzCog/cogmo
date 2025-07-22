# CogML Modular Build System

This directory contains a modular GitHub Actions workflow system based on `cogci.yml`, designed to build CogML components incrementally and handle build failures automatically.

## Architecture

### Main Dispatcher (`cogml-modular-dispatch.yml`)
The main workflow that orchestrates all builds. It can:
- Run all jobs automatically on push/PR
- Run specific jobs via `workflow_dispatch` 
- Create GitHub issues when builds fail
- Handle dependency management between jobs

### Individual Job Workflows
Each major component has its own workflow file:

| Component | File | Dependencies | Description |
|-----------|------|--------------|-------------|
| Cogutil | `cogutil-build.yml` | None | Foundation utilities |
| AtomSpace | `atomspace-build.yml` | cogutil | Core knowledge representation |
| URE | `ure-build.yml` | cogutil, atomspace | Unified Rule Engine |
| PLN | `pln-modular-build.yml` | cogutil, atomspace, ure | Probabilistic Logic Networks |
| Moses | `moses-build.yml` | cogutil | Machine Learning framework |

## Usage

### Run All Jobs
```bash
# Triggered automatically on push/PR to main branch
# Or manually trigger via GitHub Actions UI
```

### Run Specific Jobs
```bash
# Via workflow_dispatch in GitHub Actions UI:
# job_selection: "cogutil,atomspace,ure"
# cache_key_suffix: "my-test"
```

### Manual Job Execution
Each individual workflow can be run independently:
```bash
# Run just cogutil build
# Use workflow_dispatch on cogutil-build.yml
```

## Features

### 🔄 Dependency Management
- Jobs automatically wait for their dependencies
- Cache keys passed between jobs for consistency
- Automatic installation of prerequisites

### 🚨 Build Failure Handling
When builds fail, the system automatically:
1. Creates detailed GitHub issues with failure information
2. Provides remediation steps and common fixes
3. Links to the specific workflow run
4. Updates existing issues if failures persist

### 📊 Build Artifacts
Each job creates summaries with:
- Build status and timing
- Cache keys for reproducibility  
- Directory listings
- Dependency information

### 🧠 Self-Healing
- AtomSpace build includes self-healing capabilities
- Automatic retry with fixes for common issues
- Escalation to human review when needed

## Configuration

### Environment Variables
- `COGML_AUTO_FIX_ENABLED`: Enable automatic build fixes
- `COGML_MAX_FIX_ATTEMPTS`: Maximum fix attempts (default: 3)
- `COGML_ESCALATION_ENABLED`: Enable issue creation on failures

### Cache Management
- Each job produces cacheable build artifacts
- Cache keys include SHA and optional suffix for isolation
- Dependency caches automatically restored

## Development Workflow

1. **Create Feature Branch**: Work on specific component
2. **Test Individual Job**: Run specific workflow for your component
3. **Run Integration**: Use main dispatcher to test dependencies
4. **Review Issues**: Check auto-generated issues for any failures
5. **Merge**: Full pipeline runs automatically

## Extending the System

### Adding New Components
1. Create new workflow file (e.g., `newcomponent-build.yml`)
2. Add job to main dispatcher workflow
3. Update dependency chains as needed
4. Update this documentation

### Workflow Template
```yaml
---
name: CogML - Component Build
on:
  workflow_dispatch:
    inputs:
      cache_key_suffix:
        description: 'Cache key suffix'
        required: false
        default: 'default'
        type: string
  workflow_call:
    inputs:
      cache_key_suffix:
        description: 'Cache key suffix'
        required: false
        default: 'default'
        type: string

jobs:
  component:
    runs-on: blacksmith-4vcpu-ubuntu-2404
    outputs:
      cache_key: ${{ steps.cache_info.outputs.key }}
    steps:
      - uses: actions/checkout@v4
      - name: Set cache info
        id: cache_info
        run: echo "key=component-${{ runner.os }}-${{ github.sha }}-${{ inputs.cache_key_suffix }}" >> $GITHUB_OUTPUT
      # Add build steps here
```

## Troubleshooting

### Common Issues
1. **Cache Miss**: Check cache key format and dependencies
2. **Build Failures**: Review auto-generated GitHub issues
3. **Permission Issues**: Ensure workflow has required permissions

### Manual Debugging
```bash
# Check workflow runs
gh run list --workflow=cogml-modular-dispatch.yml

# View specific run details  
gh run view [run-id]

# Download artifacts
gh run download [run-id]
```

## Benefits

- **🔧 Modularity**: Build only what changed
- **🚀 Speed**: Parallel execution and caching
- **🛡️ Reliability**: Automatic issue tracking and self-healing
- **👥 Collaboration**: Clear failure reporting and remediation steps
- **📈 Scalability**: Easy to add new components