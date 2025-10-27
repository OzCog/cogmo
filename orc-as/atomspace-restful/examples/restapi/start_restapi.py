#! /usr/bin/env python3
#
# Start the restful server.
# Updated to work without full OpenCog installation

import sys
import os

# Add the parent directory to sys.path to find simple_restapi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    # Try to use the full OpenCog implementation
    from opencog.web.api.apimain import RESTAPI
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    from opencog.type_constructors import *

    # Endpoint configuration
    # To allow public access, set to 0.0.0.0; for local access, set to 127.0.0.1
    IP_ADDRESS = '0.0.0.0'
    PORT = 5000

    atomspace = AtomSpace()
    initialize_opencog(atomspace)

    Link(
        ConceptNode("Test Concept"),
        ConceptNode("another one")
    )

    api = RESTAPI(atomspace)
    api.run(host=IP_ADDRESS, port=PORT)
    
except ImportError as e:
    print(f"OpenCog not available ({e}), falling back to simple implementation...")
    
    # Use the simple implementation instead
    import importlib.util
    spec = importlib.util.spec_from_file_location("simple_restapi", 
        os.path.join(os.path.dirname(__file__), '..', '..', 'simple_restapi.py'))
    simple_restapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(simple_restapi)
    
    # The simple_restapi.py will run its own server when imported as main
    if __name__ == '__main__':
        simple_restapi.initialize_test_data()
        print("🚀 Starting AtomSpace REST API server (simple implementation)...")
        print("📍 Endpoints available at http://0.0.0.0:5000/")
        simple_restapi.app.run(host='0.0.0.0', port=5000, debug=False)
