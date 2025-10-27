__author__ = 'Cosmo Harrigan'

from flask import abort, jsonify
from flask_restful import Resource, reqparse
import socket
# Using flask-restx instead of deprecated swagger

COGSERVER_PORT = 17001


class ShellAPI(Resource):
    """
    Defines a barebones resource for sending shell commands to the CogServer
    """

    # This is because of https://github.com/twilio/flask-restful/issues/134
    @classmethod
    def new(cls, atomspace):
        cls.atomspace = atomspace
        return cls

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('command', type=str, location='args')

        super(ShellAPI, self).__init__()

    def post(self):
        """
        Send a shell command to the cogserver
        """

        # Setup socket to communicate with OpenCog CogServer
        try:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.connect(('localhost', COGSERVER_PORT))
        except socket.error as msg:
            print(msg)

        # Validate, parse and send the command
        data = reqparse.request.get_json()
        if 'command' in data:
            connection.send(data['command'])
        else:
            connection.close()
            abort(400,
                  'Invalid request: required parameter command is missing')

        connection.close()

        return jsonify({'status': 'success'})
