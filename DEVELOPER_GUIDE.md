# OpenCog Ecosystem - Developer Guide

## Quick Start

This guide helps you get started with OpenCog ecosystem development in under 30 minutes.

### Prerequisites

- **Operating System**: Ubuntu 20.04+ or similar Linux distribution
- **Python**: 3.8+ (recommended: 3.12)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 10GB+ free space for full build

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/OzCog/cogml.git
cd cogml

# Run unified setup (installs dependencies and builds core components)
./build-unified.sh --python-only

# Verify installation
python -m pytest tests/test_cognitive_primitives.py -v
```

## Development Environment Options

### Option 1: Local Development

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev python3-pip
sudo apt-get install -y libboost-all-dev librocksdb-dev guile-3.0-dev

# Install Python dependencies
pip install -r requirements-consolidated.txt

# Build core components
./build-unified.sh
```

### Option 2: Docker Development

```bash
# Build development container
docker build -t opencog-dev .

# Run development container
docker run -it -v $(pwd):/workspace opencog-dev bash
```

### Option 3: Development Container (VS Code)

If using VS Code, the repository includes devcontainer configuration:

1. Install "Remote - Containers" extension
2. Open repository in VS Code  
3. Select "Reopen in Container" when prompted
4. Development environment will be automatically configured

## Architecture Overview

The OpenCog ecosystem is organized into 13 component categories:

### Core Components (orc-as, orc-dv)
- **AtomSpace**: Hypergraph knowledge representation
- **CogUtil**: Core utilities and data structures

### AI & Learning (orc-ai)
- **PLN**: Probabilistic Logic Network
- **URE**: Unified Rule Engine
- **Learn**: Structure learning algorithms
- **Moses**: Meta-optimizing semantic evolutionary search

### Natural Language (orc-nl)  
- **Link Grammar**: Natural language parsing
- **RelEx**: Relationship extraction

### Robotics (orc-ro)
- **Sensory**: Sensor data processing
- **Perception**: Computer vision and sensory integration

### Web & APIs (orc-wb, orc-sv)
- **CogServer**: Network API server
- **REST APIs**: HTTP interfaces
- **WebSocket APIs**: Real-time communication

## Component Development

### Adding a New Component

1. **Create component directory**:
   ```bash
   mkdir orc-{category}/{component-name}
   cd orc-{category}/{component-name}
   ```

2. **Add build configuration**:
   ```bash
   # For CMake components
   touch CMakeLists.txt
   
   # For Python components  
   touch setup.py
   
   # For Rust components
   cargo init
   ```

3. **Add documentation**:
   ```bash
   touch README.md
   ```

4. **Update build order** in `build-unified.sh` if needed

### Component Structure

```
orc-{category}/{component}/
├── CMakeLists.txt          # CMake build configuration
├── setup.py                # Python package configuration  
├── Cargo.toml             # Rust package configuration
├── README.md              # Component documentation
├── src/                   # Source code
├── tests/                 # Unit tests
├── examples/              # Usage examples
└── docs/                  # Additional documentation
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_cognitive_primitives.py -v

# Run tests with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific component tests
./build-unified.sh --test as/atomspace
```

### Writing Tests

Follow pytest conventions:

```python
# tests/test_my_component.py
import pytest
from my_component import MyClass

class TestMyComponent:
    def test_basic_functionality(self):
        obj = MyClass()
        assert obj.method() == expected_result
        
    def test_error_handling(self):
        obj = MyClass()
        with pytest.raises(ValueError):
            obj.invalid_operation()
```

## Code Quality

### Automated Formatting

```bash
# Format Python code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy .
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
pip install pre-commit
pre-commit install
```

## Debugging

### Python Debugging

```python
# Add debugging breakpoint
import pdb; pdb.set_trace()

# Or use ipdb for better interface  
import ipdb; ipdb.set_trace()
```

### CMake Debugging

```bash
# Verbose build output
make VERBOSE=1

# Debug CMake configuration
cmake -DCMAKE_VERBOSE_MAKEFILE=ON .
```

### Performance Profiling

```bash
# Profile Python code
python -m cProfile -o profile.stats script.py

# Analyze profile
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

## Component Integration

### AtomSpace Integration

```python
from opencog.atomspace import AtomSpace, types
from opencog.type_constructors import *

# Create AtomSpace
atomspace = AtomSpace()

# Add atoms
concept_node = ConceptNode("example")
atomspace.add_atom(concept_node)

# Query atoms
atoms = atomspace.get_atoms_by_type(types.ConceptNode)
```

### CogServer Integration

```python
from opencog.cogserver import CogServer

# Connect to CogServer
cogserver = CogServer()
cogserver.connect("localhost", 17001)

# Send commands
result = cogserver.send("(+ 2 3)")
print(result)
```

## Performance Optimization

### Memory Management

```python
# Use context managers for AtomSpace
with AtomSpace() as atomspace:
    # Atoms are automatically cleaned up
    pass

# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

### Parallel Processing

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# CPU-intensive tasks
def process_data(data_chunk):
    # Process data
    return result

# Parallel processing
with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
    futures = [executor.submit(process_data, chunk) for chunk in data_chunks]
    results = [future.result() for future in futures]
```

## Contribution Guidelines

### Pull Request Process

1. **Fork and branch**: Create feature branch from `develop`
2. **Develop**: Make changes following code style guidelines
3. **Test**: Ensure all tests pass and add new tests if needed
4. **Document**: Update documentation for any API changes
5. **Pull Request**: Submit PR with clear description

### Commit Messages

Follow conventional commit format:

```
type(scope): description

feat(atomspace): add new query functionality
fix(cogserver): resolve connection timeout issue  
docs(readme): update installation instructions
test(pln): add unit tests for reasoning engine
```

### Code Review

- **All PRs require review** from at least one maintainer
- **Automated checks must pass** (CI/CD, quality checks)
- **Documentation updates** required for API changes
- **Performance impact** should be considered for core components

## Troubleshooting

### Common Issues

**Build failures**:
```bash
# Clean build and try again
./build-unified.sh --clean

# Check system dependencies
apt list --installed | grep -E "(boost|cmake|python3-dev)"
```

**Import errors**:
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Install in development mode
pip install -e .
```

**Memory issues**:
```bash
# Monitor memory during builds
watch -n 1 'free -h && ps aux | grep -E "(cmake|make|python)" | head -5'

# Reduce parallel jobs
export PARALLEL_JOBS=2
./build-unified.sh
```

### Getting Help

- **Documentation**: Check component-specific README files
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join OpenCog community forums

## Performance Benchmarking

### Running Benchmarks

```bash
# Run performance benchmarks
python -m pytest benchmarks/ --benchmark-json=results.json

# Compare benchmark results
python -m benchmarks.compare_results baseline.json current.json
```

### Creating Benchmarks

```python
# benchmarks/test_performance.py
import pytest

@pytest.mark.benchmark(group="atomspace")
def test_atom_creation_speed(benchmark):
    def create_atoms():
        atomspace = AtomSpace()
        for i in range(1000):
            ConceptNode(f"node_{i}")
    
    result = benchmark(create_atoms)
```

## Advanced Topics

### Custom AtomSpace Types

```cpp
// Define custom atom types in C++
#define CUSTOM_TYPES \
    CUSTOM_NODE(CustomNode, NODE) \
    CUSTOM_LINK(CustomLink, LINK)

// Register types
void register_custom_types() {
    opencog::classserver().addType(CUSTOM_NODE, "CustomNode");
}
```

### Distributed Processing  

```python
# Set up distributed AtomSpace
from opencog.cogserver import DistributedAtomSpace

das = DistributedAtomSpace()
das.connect_cluster(["node1:17001", "node2:17001"])
```

This developer guide provides the essential information needed to contribute effectively to the OpenCog ecosystem.