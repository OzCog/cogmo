__author__ = 'Cosmo Harrigan'

from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS
from opencog.web.api.apiatomcollection import *
from opencog.web.api.apitypes import *
from opencog.web.api.apishell import *
from opencog.web.api.apischeme import *
from opencog.web.api.apighost import *
# Using flask-restx instead of deprecated flask-restful-swagger
from flask_restx import Api, Resource, fields


class RESTAPI(object):
    """
    REST API for OpenCog

    Implemented using the Flask micro-framework and Flask-RESTful extension

    Documentation:
    http://wiki.opencog.org/w/REST_API

    Prerequisites:
    Flask, flask-restful, flask-cors, flask-restx

    Default endpoint: http://127.0.0.1:5000/api/v1.1/
    (Replace 127.0.0.1 with the IP address of the server if necessary)

    Example request: http://127.0.0.1:5000/api/v1.1/atoms?type=ConceptNode

    See: opencog/python/web/api/exampleclient.py for detailed examples of
    usage, and review the method definitions in each resource for request/
    response specifications.
    """

    def __init__(self, atomspace):
        self.atomspace = atomspace

        # Initialize the web server and set the routing
        self.app = Flask(__name__, static_url_path="")
        self.api = Api(self.app, version='1.1', title='AtomSpace REST API', 
                      description='RESTful interface to OpenCog AtomSpace',
                      doc='/api/v1.1/')

        # Allow Cross Origin Resource Sharing (CORS) so that javascript apps
        # can use this API from other domains, ports and protocols. 
        self.cors = CORS(self.app, resources={r"/api/*": {"origins": "*"}})

        # Create and add each resource
        atom_collection_api = AtomCollectionAPI.new(self.atomspace)
        atom_types_api = TypesAPI
        shell_api = ShellAPI
        scheme_api = SchemeAPI.new(self.atomspace)
        ghost_api = GhostApi.new(self.atomspace)

        # CORS is already handled by CORS(self.app) above

        self.api.add_resource(atom_collection_api,
                              '/api/v1.1/atoms',
                              '/api/v1.1/atoms/<int:id>', endpoint='atoms')
        self.api.add_resource(atom_types_api,
                              '/api/v1.1/types',
                              endpoint='types')
        self.api.add_resource(shell_api,
                              '/api/v1.1/shell',
                              endpoint='shell')
        self.api.add_resource(scheme_api,
                              '/api/v1.1/scheme',
                              endpoint='scheme')
        self.api.add_resource(ghost_api, 
                              '/api/v1.1/ghost', 
                              endpoint='ghost')


    def run(self, host='127.0.0.1', port=5000):
        """
        Runs the REST API

        :param host: the hostname to listen on. Set this to ``'0.0.0.0'`` to
                     have the server available externally as well. Defaults to
                     ``'127.0.0.1'``.
        :param port: the port of the webserver. Defaults to ``5000``
        """
        self.app.run(debug=False, host=host, port=port)

    def test(self):
        """
        Returns a test client for the REST API
        """
        return self.app.test_client()
