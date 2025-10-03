# AtomSpace RESTful Web API - Updated Implementation

## Overview

This module provides a modern, working RESTful web interface to the AtomSpace. The implementation has been updated to address all the issues mentioned in the original README.

## What's New

### ✅ Fixed Issues
1. **Removed deprecated dependencies**: Replaced `flask-restful-swagger` with `flask-restx`
2. **Removed obsolete AtomSpace APIs**: No longer uses deprecated attention bank functions
3. **Ubuntu 22.04 compatible**: Works with current Python and Flask versions
4. **Correct JSON format**: Proper representation of AtomSpace contents
5. **Working implementation**: Fully functional REST API server

### ✅ Modern Dependencies
- Flask 3.1+
- flask-restful 0.3+
- flask-cors 6.0+
- flask-restx 1.3+ (instead of deprecated swagger)

## Quick Start

### Prerequisites
```bash
pip3 install flask flask-restful flask-cors flask-restx
```

### Running the Server

#### Option 1: Simple Implementation (Recommended)
```bash
cd orc-as/atomspace-restful
python3 simple_restapi.py
```

#### Option 2: Integrated Version
```bash
cd orc-as/atomspace-restful/examples/restapi
python3 start_restapi.py
```

Both will start a server at `http://127.0.0.1:5000`

## API Endpoints

### Core Endpoints
- `GET /` - API information and endpoint list
- `GET /api/v1.1/atoms` - Get all atoms
- `GET /api/v1.1/atoms?type=ConceptNode` - Filter atoms by type
- `GET /api/v1.1/atoms?name=animal` - Filter atoms by name
- `GET /api/v1.1/atoms/<id>` - Get specific atom
- `POST /api/v1.1/atoms` - Create new atom
- `DELETE /api/v1.1/atoms/<id>` - Delete atom
- `GET /api/v1.1/types` - Get available atom types
- `GET /api/v1.1/stats` - AtomSpace statistics
- `GET /api/v1.1/validate` - Validate AtomSpace integrity

### Example Usage

#### Get all atoms
```bash
curl http://127.0.0.1:5000/api/v1.1/atoms
```

#### Create a new ConceptNode
```bash
curl -X POST http://127.0.0.1:5000/api/v1.1/atoms \
  -H "Content-Type: application/json" \
  -d '{
    "type": "ConceptNode",
    "name": "robot",
    "truthvalue": {
      "details": {
        "strength": 0.9,
        "confidence": 0.8
      }
    }
  }'
```

#### Get atoms by type
```bash
curl "http://127.0.0.1:5000/api/v1.1/atoms?type=ConceptNode"
```

#### Get statistics
```bash
curl http://127.0.0.1:5000/api/v1.1/stats
```

## JSON Format

### Atom Representation
```json
{
  "id": "uuid-string",
  "type": "ConceptNode",
  "name": "animal",
  "outgoing": [],
  "incoming": ["uuid-of-incoming-link"],
  "truthvalue": {
    "type": "simple",
    "details": {
      "strength": 1.0,
      "confidence": 1.0
    }
  }
}
```

### Response Format
```json
{
  "result": {
    "complete": true,
    "skipped": 0,
    "total": 7,
    "atoms": [...]
  }
}
```

## Architecture

### Simple Implementation (`simple_restapi.py`)
- Self-contained Flask application
- In-memory AtomSpace simulation
- No external OpenCog dependencies
- Perfect for development and testing

### Integration Points
- Backward compatible with existing AtomSpace Explorer
- CORS enabled for web applications
- JSONP callback support
- Proper error handling and HTTP status codes

## Testing

The implementation passes all hypergraph API tests:
```bash
./hypergraph-api-test.sh
```

Expected output:
```
✅ API ENDPOINT STRUCTURE VALIDATED
✅ HYPERGRAPH API TESTING COMPLETE
```

## Development

### Adding New Endpoints
1. Add new Resource class in `simple_restapi.py`
2. Register with `api.add_resource()`
3. Test with curl or web client

### Integration with Full OpenCog
The `start_restapi.py` automatically detects if OpenCog is available and falls back to the simple implementation if not.

## Compatibility

- ✅ Python 3.8+
- ✅ Ubuntu 20.04+, 22.04+
- ✅ Modern Flask ecosystem
- ✅ Web browsers with CORS
- ✅ AtomSpace Explorer integration

## Migration from Old Version

If you were using the old deprecated version:
1. Update your Python dependencies (remove flask-restful-swagger)
2. Use the new `simple_restapi.py` or updated `start_restapi.py`
3. No API changes needed - endpoints remain the same
4. JSON format is now correct and consistent