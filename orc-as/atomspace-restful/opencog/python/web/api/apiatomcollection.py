__author__ = 'Cosmo Harrigan'

from flask import abort, json, current_app, jsonify
from flask_restful import Resource, reqparse, marshal
# import opencog.cogserver  # Removed - not used
from opencog.atomspace import Atom
from opencog.web.api.mappers import *
from flask_restful.utils import cors

# Using flask-restx instead of deprecated swagger
from flask_restx import Namespace, fields
# Removed deprecated AttentionBank import

# Temporary hack
from opencog.web.api.utilities import get_atoms_by_name

# If the system doesn't have these dependencies installed, display a warning
# but allow the API to load
try:
    from graphviz import dot
except ImportError:
    print ("DOT graph description format option not enabled in REST API. To " \
          "enable, install the dependencies listed here:\n" \
          "https://github.com/opencog/opencog/tree/master/opencog/python/graph_description#prerequisites")

"AtomSpace management functionality"
class AtomCollectionAPI(Resource):
    # This is because of https://github.com/twilio/flask-restful/issues/134
    @classmethod
    def new(cls, atomspace):
        cls.atomspace = atomspace
        return cls

    def __init__(self):
        self.atom_map = global_atom_map
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('type', type=str, action='append',
                     location='args', choices=types.__dict__.keys())
        self.reqparse.add_argument('name', type=str, location='args')
        self.reqparse.add_argument('callback', type=str, location='args')
        self.reqparse.add_argument('filterby', type=str, location='args',
                                   choices=[])
        self.reqparse.add_argument('stimin', type=int, location='args')
        self.reqparse.add_argument('stimax', type=int, location='args')
        self.reqparse.add_argument('tvStrengthMin', type=float, location='args')
        self.reqparse.add_argument(
            'tvConfidenceMin', type=float, location='args')
        self.reqparse.add_argument('tvCountMin', type=float, location='args')
        self.reqparse.add_argument(
            'includeIncoming', type=str, location='args',
            choices=['true', 'false', 'True', 'False', '0', '1'])
        self.reqparse.add_argument(
            'includeOutgoing', type=str, location='args',
            choices=['true', 'false', 'True', 'False', '0', '1'])
        self.reqparse.add_argument(
            'dot', type=str, location='args',
            choices=['true', 'false', 'True', 'False', '0', '1'])
        self.reqparse.add_argument('limit', type=int, location='args')

        super(AtomCollectionAPI, self).__init__()
        # self.atomspace = opencog.cogserver.get_server_atomspace()

    # Set CORS headers to allow cross-origin access
    # (https://github.com/twilio/flask-restful/pull/131):
    @cors.crossdomain(origin='*')

    def get(self, id=""):
        retval = jsonify({'error':'Internal error'})
        try:
           retval = self._get(id=id)
        except Exception as e:
           retval = jsonify({'error':str(e)})
        return retval

    def _get(self, id=""):
        """
        Returns a list of atoms matching the specified criteria
        """

        args = self.reqparse.parse_args()
        type = args.get('type')
        name = args.get('name')
        callback = args.get('callback')

        filter_by = args.get('filterby')
        sti_min = args.get('stimin')
        sti_max = args.get('stimax')

        tv_strength_min = args.get('tvStrengthMin')
        tv_confidence_min = args.get('tvConfidenceMin')
        tv_count_min = args.get('tvCountMin')

        include_incoming = args.get('includeIncoming')
        include_outgoing = args.get('includeOutgoing')

        dot_format = args.get('dot')

        limit = args.get('limit')

        if id != "":
            try:
                atom = self.atom_map.get_atom(int(id))
                atoms = [atom]
            except IndexError:
                atoms = []
                # abort(404, 'Atom not found')
        else:
            # First, check if there is a valid filter type, and give it
            # precedence if it exists
            valid_filter = False
            if filter_by is not None:
                if filter_by == 'stirange':
                    if sti_min is not None:
                        valid_filter = True
                        # Deprecated: get_atoms_by_av() no longer available
                        # Return empty list for now
                        atoms = []
                    else:
                        abort(400, 'Invalid request: stirange filter requires '
                                   'stimin parameter')
                elif filter_by == 'attentionalfocus':
                    valid_filter = True
                    # Deprecated: get_atoms_in_attentional_focus() no longer available
                    # Return empty list for now
                    atoms = []

            # If there is not a valid filter type, proceed to select by type
            # or name
            if not valid_filter:
                if type is None and name is None:
                    atoms = self.atomspace.get_atoms_by_type(types.Atom)
                elif name is None:
                    atoms = []
                    for t in type:
                         atoms = atoms + self.atomspace.get_atoms_by_type(
                                           types.__dict__.get(t))
                else:
                    if type is None:
                        type = ['Node']
                    for t in type:
                        atoms = list(get_atoms_by_name(types.__dict__.get(t),
                                    name, self.atomspace))

            # Optionally, filter by TruthValue
            if tv_strength_min is not None:
                atoms = [atom for atom in atoms if atom.tv.mean >=
                                                   tv_strength_min]

            if tv_confidence_min is not None:
                atoms = [atom for atom in atoms if atom.tv.confidence >=
                                                   tv_confidence_min]

            if tv_count_min is not None:
                atoms = [atom for atom in atoms if atom.tv.count >=
                                                   tv_count_min]

        # Optionally, include the incoming set
        if include_incoming in ['True', 'true', '1']:
            atoms = self.atomspace.include_incoming(atoms)

        # Optionally, include the outgoing set
        if include_outgoing in ['True', 'true', '1']:
            atoms = self.atomspace.include_outgoing(atoms)

        # Optionally, limit number of atoms returned
        if limit is not None:
            if len(atoms) > limit:
                atoms = atoms[0:limit]

        # The default is to return the atom set as JSON atoms. Optionally, a
        # DOT return format is also supported
        if dot_format not in ['True', 'true', '1']:
            atom_list = AtomListResponse(atoms)
            # xxxxxxxxxxxx here add atoms
            json_data = {'result': atom_list.format()}

            # if callback function supplied, pad the JSON data (i.e. JSONP):
            if callback is not None:
                response = str(callback) + '(' + json.dumps(json_data) + ');'
                return current_app.response_class(
                    response, mimetype='application/javascript')
            else:
                return current_app.response_class(
                    json.dumps(json_data), mimetype='application/json')
        else:
            dot_output = dot.get_dot_representation(atoms)
            return jsonify({'result': dot_output})

    def post(self):
        """
        Creates a new atom. If the atom already exists, it updates the atom.
        """

        # Prepare the atom data and validate it
        data = reqparse.request.get_json()

        if 'type' in data:
            if data['type'] in types.__dict__:
                type = types.__dict__.get(data['type'])
            else:
                abort(400, 'Invalid request: type \'' + type + '\' is not a '
                                                               'valid type')
        else:
            abort(400, 'Invalid request: required parameter type is missing')

        # TruthValue
        tv = ParseTruthValue.parse(data)

        # Outgoing set
        if 'outgoing' in data:
            print (data)
            if len(data['outgoing']) > 0:
                outgoing = [self.atom_map.get_atom(uid)
                                for uid in data['outgoing']]
        else:
            outgoing = None

        # Name
        name = data['name'] if 'name' in data else None

        # Nodes must have names
        if is_a(type, types.Node):
            if name is None:
                abort(400, 'Invalid request: node type specified and required '
                           'parameter name is missing')
        # Links can't have names
        else:
            if name is not None:
                abort(400, 'Invalid request: parameter name is not allowed for '
                           'link types')

        try:
            atom = self.atomspace.add(t=type, name=name, tv=tv, out=outgoing)
            uid = self.atom_map.get_uid(atom)
        except TypeError:
            abort(500, 'Error while processing your request. Check your '
                       'parameters.')

        dictoid = marshal(atom, atom_fields)
        dictoid['handle'] = uid
        return {'atoms': dictoid}

    def put(self, id):
        """
        Updates the AttentionValue (STI, LTI, VLTI) or TruthValue of an atom
        """

        # If the atom is not found in the atomspace.
        the_atom = self.atom_map.get_atom(id)
        if the_atom is not None:
            # Prepare the atom data
            data = reqparse.request.get_json()

            if 'truthvalue' not in data and 'attentionvalue' not in data:
                abort(400, 'Invalid request: you must include a truthvalue or '
                           'attentionvalue parameter')

            if 'truthvalue' in data:
                tv = ParseTruthValue.parse(data)
                the_atom.tv = tv

            if 'attentionvalue' in data:
                (sti, lti, vlti) = ParseAttentionValue.parse(data)				                
                attention_bank = AttentionBank(the_atom.atomspace)
                attention_bank.set_av(the_atom, sti, lti)

            dicty = marshal(the_atom, atom_fields)
            dicty['handle'] = self.atom_map.get_uid(the_atom)
            if 'attentionvalue' in data:
                dicty['attentionvalue'] = {
                    'sti': attention_bank.get_sti(the_atom),
                    'lti': attention_bank.get_lti(the_atom),
                    'vlti': attention_bank.get_vlti(the_atom)
                }
            return {'atoms': dicty}
        else:
            abort(404, 'Atom not found')

    def delete(self, id):
        """
        Removes an atom from the AtomSpace
        """

        atom = self.atom_map.get_atom(id)
        if atom is not None:
            status = self.atomspace.remove(atom)
            self.atom_map.remove(atom, id)
            response = DeleteAtomResponse(id, status)
            return {'result': response.format()}
        else:
            abort(404, 'Atom not found')
