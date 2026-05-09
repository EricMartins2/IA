from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

print('Iniciando...')

lingua = input('Para qual lingua você quer traduzir? ')
texto = input('Digite o texto para a \033[1;34mIA\033[m traduzir: ')

mensagens = [
    SystemMessage(f'Traduza o texto a seguir para {lingua}'),
    HumanMessage(f'{texto}')
]

modelo = ChatOpenAI(model="gpt-5.4-mini")
parser = StrOutputParser()
cadeia = modelo | parser

extraction = cadeia.invoke(mensagens)
print(f'Tradução: {extraction}')

