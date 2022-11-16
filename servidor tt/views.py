from flask import Blueprint, jsonify, redirect, url_for, request
from clasificador import clasificar_texto_agresivo, clasificar_texto_ofensivo, clasificar_texto_vulgar
views = Blueprint(__name__, "views")


@views.route("/")
def home():
    return "home page"


@views.route("/clasificar", methods=['POST', 'GET'])
def clasificar():
    if request.method == 'POST':
        print("CONTENIDO PARA CLASIFICAR")
        resultados = {'clasificacion': []}
        content = request.get_json(silent=True)
        aux = content['data']
        for texto in aux["contenido"]:
            resultados["clasificacion"].append([clasificar_texto_vulgar(
                texto), clasificar_texto_agresivo(texto), clasificar_texto_ofensivo(texto), texto])

        return jsonify(resultados)
