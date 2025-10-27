__author__ = 'Cosmo Harrigan'

from flask import abort, jsonify
from flask_restful import Resource, reqparse
from opencog.scheme_wrapper import scheme_eval, __init__
# Using flask-restx instead of deprecated swagger

COGSERVER_PORT = 17001


class SchemeAPI(Resource):
    """
    Defines an interface for issuing commands to and receiving responses from
    the OpenCog Scheme interpreter
    """

    # This is because of https://github.com/twilio/flask-restful/issues/134
    @classmethod
    def new(cls, atomspace):
        cls.atomspace = atomspace
        return cls

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('command', type=str, location='args')

        super(SchemeAPI, self).__init__()

    def post(self):
        """
        Send a command to the Scheme interpreter
        """

        # Validate, parse and send the command
        data = reqparse.request.get_json()
        if 'command' in data:
            response = scheme_eval(self.atomspace, data['command'])
        else:
            abort(400,
                  'Invalid request: required parameter command is missing')

        return jsonify({'response': response})
