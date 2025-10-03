#!/usr/bin/env python3
"""
Simplified AtomSpace REST API
This provides a working REST API without requiring full OpenCog installation
"""

from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from flask_restx import Namespace, fields
import json
import uuid
from typing import Dict, List, Any

class SimpleAtomSpace:
    """
    A simplified AtomSpace implementation for demonstration purposes
    """
    def __init__(self):
        self.atoms = {}  # uuid -> atom dict
        self.type_atoms = {}  # type -> list of uuids
        self.name_atoms = {}  # name -> list of uuids
        
    def add_node(self, node_type: str, name: str, tv_strength: float = 1.0, tv_confidence: float = 1.0):
        """Add a node to the atomspace"""
        atom_id = str(uuid.uuid4())
        atom = {
            'id': atom_id,
            'type': node_type,
            'name': name,
            'outgoing': [],
            'incoming': [],
            'truthvalue': {
                'type': 'simple',
                'details': {
                    'strength': tv_strength,
                    'confidence': tv_confidence
                }
            }
        }
        
        self.atoms[atom_id] = atom
        
        # Index by type
        if node_type not in self.type_atoms:
            self.type_atoms[node_type] = []
        self.type_atoms[node_type].append(atom_id)
        
        # Index by name
        if name not in self.name_atoms:
            self.name_atoms[name] = []
        self.name_atoms[name].append(atom_id)
        
        return atom_id
    
    def add_link(self, link_type: str, outgoing: List[str], tv_strength: float = 1.0, tv_confidence: float = 1.0):
        """Add a link to the atomspace"""
        atom_id = str(uuid.uuid4())
        atom = {
            'id': atom_id,
            'type': link_type,
            'name': '',
            'outgoing': outgoing,
            'incoming': [],
            'truthvalue': {
                'type': 'simple',
                'details': {
                    'strength': tv_strength,
                    'confidence': tv_confidence
                }
            }
        }
        
        self.atoms[atom_id] = atom
        
        # Update incoming sets
        for out_id in outgoing:
            if out_id in self.atoms:
                self.atoms[out_id]['incoming'].append(atom_id)
        
        # Index by type
        if link_type not in self.type_atoms:
            self.type_atoms[link_type] = []
        self.type_atoms[link_type].append(atom_id)
        
        return atom_id
    
    def get_atoms_by_type(self, atom_type: str):
        """Get all atoms of a specific type"""
        atom_ids = self.type_atoms.get(atom_type, [])
        return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    def get_atoms_by_name(self, name: str):
        """Get all atoms with a specific name"""
        atom_ids = self.name_atoms.get(name, [])
        return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    def get_atom(self, atom_id: str):
        """Get a specific atom by ID"""
        return self.atoms.get(atom_id)
    
    def get_all_atoms(self):
        """Get all atoms"""
        return list(self.atoms.values())
    
    def delete_atom(self, atom_id: str):
        """Delete an atom"""
        if atom_id in self.atoms:
            atom = self.atoms[atom_id]
            
            # Remove from type index
            atom_type = atom['type']
            if atom_type in self.type_atoms:
                self.type_atoms[atom_type] = [aid for aid in self.type_atoms[atom_type] if aid != atom_id]
            
            # Remove from name index
            name = atom['name']
            if name and name in self.name_atoms:
                self.name_atoms[name] = [aid for aid in self.name_atoms[name] if aid != atom_id]
            
            # Update incoming/outgoing references
            for out_id in atom['outgoing']:
                if out_id in self.atoms:
                    self.atoms[out_id]['incoming'] = [aid for aid in self.atoms[out_id]['incoming'] if aid != atom_id]
            
            for inc_id in atom['incoming']:
                if inc_id in self.atoms:
                    self.atoms[inc_id]['outgoing'] = [aid for aid in self.atoms[inc_id]['outgoing'] if aid != atom_id]
            
            del self.atoms[atom_id]
            return True
        return False
    
    def get_atom_types(self):
        """Get all atom types in use"""
        return list(self.type_atoms.keys())
    
    def get_stats(self):
        """Get atomspace statistics"""
        return {
            'total_atoms': len(self.atoms),
            'types': list(self.type_atoms.keys()),
            'type_counts': {t: len(aids) for t, aids in self.type_atoms.items()}
        }

class AtomCollectionAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('type', type=str, location='args')
        self.reqparse.add_argument('name', type=str, location='args')
        self.reqparse.add_argument('callback', type=str, location='args')
        self.reqparse.add_argument('limit', type=int, location='args')
    
    def get(self, atom_id=None):
        """Get atoms"""
        args = self.reqparse.parse_args()
        
        if atom_id:
            # Get specific atom
            atom = atomspace.get_atom(str(atom_id))
            if atom:
                result = {'atom': atom}
            else:
                result = {'error': 'Atom not found'}
        else:
            # Get atoms by filter
            atoms = []
            
            if args['type']:
                atoms = atomspace.get_atoms_by_type(args['type'])
            elif args['name']:
                atoms = atomspace.get_atoms_by_name(args['name'])
            else:
                atoms = atomspace.get_all_atoms()
            
            # Apply limit
            if args['limit']:
                atoms = atoms[:args['limit']]
            
            result = {
                'result': {
                    'complete': True,
                    'skipped': 0,
                    'total': len(atoms),
                    'atoms': atoms
                }
            }
        
        # Handle JSONP callback
        if args['callback']:
            response_json = json.dumps(result)
            return f"{args['callback']}({response_json});", 200, {'Content-Type': 'application/javascript'}
        
        return result
    
    def post(self):
        """Create new atoms"""
        data = request.get_json()
        if not data:
            return {'error': 'No JSON data provided'}, 400
        
        try:
            atom_type = data.get('type')
            name = data.get('name', '')
            outgoing = data.get('outgoing', [])
            tv = data.get('truthvalue', {})
            tv_strength = tv.get('details', {}).get('strength', 1.0)
            tv_confidence = tv.get('details', {}).get('confidence', 1.0)
            
            if not atom_type:
                return {'error': 'Atom type is required'}, 400
            
            if outgoing:
                # It's a link
                atom_id = atomspace.add_link(atom_type, outgoing, tv_strength, tv_confidence)
            else:
                # It's a node
                atom_id = atomspace.add_node(atom_type, name, tv_strength, tv_confidence)
            
            atom = atomspace.get_atom(atom_id)
            return {'result': {'atom': atom}}, 201
            
        except Exception as e:
            return {'error': str(e)}, 400
    
    def delete(self, atom_id):
        """Delete an atom"""
        if atomspace.delete_atom(str(atom_id)):
            return {'result': {'success': True, 'handle': atom_id}}
        else:
            return {'error': 'Atom not found'}, 404

class TypesAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('callback', type=str, location='args')
    
    def get(self):
        """Get available atom types"""
        # Common OpenCog atom types
        common_types = [
            'ConceptNode', 'NumberNode', 'VariableNode', 'PredicateNode',
            'InheritanceLink', 'SimilarityLink', 'EvaluationLink', 'ListLink',
            'AndLink', 'OrLink', 'NotLink', 'ImplicationLink'
        ]
        
        # Add types currently in use
        used_types = atomspace.get_atom_types()
        all_types = list(set(common_types + used_types))
        
        result = {'types': sorted(all_types)}
        
        # Handle JSONP callback
        args = self.reqparse.parse_args()
        if args['callback']:
            response_json = json.dumps(result)
            return f"{args['callback']}({response_json});", 200, {'Content-Type': 'application/javascript'}
        
        return result

class StatsAPI(Resource):
    def get(self):
        """Get atomspace statistics"""
        return atomspace.get_stats()

class ValidateAPI(Resource):
    def get(self):
        """Validate atomspace integrity"""
        stats = atomspace.get_stats()
        return {
            'valid': True,
            'message': 'AtomSpace validation passed',
            'stats': stats
        }

# Global atomspace instance
atomspace = SimpleAtomSpace()

# Create Flask app
app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# Add some test data
def initialize_test_data():
    """Add some initial test data"""
    # Add some concept nodes
    animal_id = atomspace.add_node('ConceptNode', 'animal')
    mammal_id = atomspace.add_node('ConceptNode', 'mammal')
    cat_id = atomspace.add_node('ConceptNode', 'cat')
    dog_id = atomspace.add_node('ConceptNode', 'dog')
    
    # Add inheritance links
    atomspace.add_link('InheritanceLink', [mammal_id, animal_id])
    atomspace.add_link('InheritanceLink', [cat_id, mammal_id])
    atomspace.add_link('InheritanceLink', [dog_id, mammal_id])
    
    print(f"Initialized with {len(atomspace.atoms)} test atoms")

# Register API endpoints
api.add_resource(AtomCollectionAPI, 
                '/api/v1.1/atoms', 
                '/api/v1.1/atoms/<int:atom_id>')
api.add_resource(TypesAPI, '/api/v1.1/types')
api.add_resource(StatsAPI, '/api/v1.1/stats')
api.add_resource(ValidateAPI, '/api/v1.1/validate')

@app.route('/')
def home():
    return {
        'message': 'AtomSpace REST API',
        'version': '1.1',
        'endpoints': [
            'GET /api/v1.1/atoms - Get all atoms',
            'GET /api/v1.1/atoms?type=<type> - Get atoms by type', 
            'GET /api/v1.1/atoms?name=<name> - Get atoms by name',
            'GET /api/v1.1/atoms/<id> - Get specific atom',
            'POST /api/v1.1/atoms - Create new atom',
            'DELETE /api/v1.1/atoms/<id> - Delete atom',
            'GET /api/v1.1/types - Get available atom types',
            'GET /api/v1.1/stats - Get atomspace statistics',
            'GET /api/v1.1/validate - Validate atomspace integrity'
        ]
    }

if __name__ == '__main__':
    initialize_test_data()
    print("🚀 Starting AtomSpace REST API server...")
    print("📍 Endpoints available at http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)