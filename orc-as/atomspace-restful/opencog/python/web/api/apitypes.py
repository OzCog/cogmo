__author__ = 'Cosmo Harrigan'

from flask import json, current_app
from flask_restful import Resource, reqparse
from opencog.web.api.mappers import *
from flask_restful.utils import cors
# Using flask-restx instead of deprecated swagger

class TypesAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('callback', type=str, location='args')
        super(TypesAPI, self).__init__()

    # Set CORS headers to allow cross-origin access
    # (https://github.com/twilio/flask-restful/pull/131):
    @cors.crossdomain(origin='*')
    def get(self):
        """
        Returns a list of valid atom types
        """

        json_data = \
            {'types': filter(lambda x:
                             not x.startswith('__') and not x.endswith('__')
                             and not x == 'NO_TYPE', types.__dict__.keys())}

        # if callback function supplied, pad the JSON data (i.e. JSONP):
        args = self.reqparse.parse_args()
        callback = args.get('callback')
        if callback is not None:
            response = str(callback) + '(' + json.dumps(json_data) + ');'
            return current_app.response_class(
                response, mimetype='application/javascript')
        else:
            return current_app.response_class(
                json.dumps(json_data), mimetype='application/json')
