from pytubefix import YouTube
import ffmpeg

url = "https://www.youtube.com/shorts/ac7XgLVurz4"


yt = YouTube(url)


ys = yt.streams.get_highest_resolution().url # Com base na lista de formatos(streams), por exemplo, 720p, 1080p etc., escolha a de melhor qualidade de resolução



ffmpeg.input(ys).output('audio.mp3').run()


import whisperx
from whisperx.diarize import DiarizationPipeline
import os

#Transcrição do audio
modelo = whisperx.load_model('large-v2', device='cpu') # -> Definição do modelo de escuta do audio, nível de inteligência e onde vai rodar
audio = whisperx.load_audio("audio.mp3") # -> Pega o arquivo de aúdio/vídeo
result = modelo.transcribe(audio) # Transforma o aúdio em texto
print('Trascrição:', result['segments'])

#Detecção dos falantes
modelo_diarizacao = DiarizationPipeline( # -> DiarizationPipeline: Detecta quem está falando
    token='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', device='cpu') # -> O token é a chave para o DiarizationPipeline acessar e escolher o modelo de diarização (separar os falantes)

segmentos_diarizacao = modelo_diarizacao(audio) # -> Especifiquei o audio no qual o modelo de diarização deve trabalhar

resultado = whisperx.assign_word_speakers(segmentos_diarizacao, result) # -> Juntei os falantes com seu
print(segmentos_diarizacao) # -> Imprimi os falantes
print(resultado['segments']) # Imprimi as falas com os respectivos falantes no final de cada parte


#Estruturando os falantes com suas falas
# -> Cada linha de segmento contém o 'text:' e um 'speaker'
for segmento in resultado['segments']: # -> Vai passar por cada segmento
    falante = segmento['speaker'] # -> A cada nova repetição, um novo 'speaker' vai ser atribuído ao falante
    texto = segmento['text'] # -> A cada nova repetição, um novo 'text' vai ser atribuído ao texto
    final_transcricao = '{}: {}'.format(falante, texto) # Organizei a estrutura da resposta de cada repetição
    print(final_transcricao)

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from time import sleep



template = '''
    Você é um assistente para resumir Podcasts.

    Você receberá o conteúdo transcrito de um Podcast e deverá
    criar um resumo, em MARKDOWN, contendo: 
    1. Uma introdução
        - Escreva 1 parágrafo com uma introdução sobre o que foi falado no podcast
    
    2. Os principais pontos abordados, em ordem cronológica. Numere-os.
        - Escreva um texto bem completo sobre cada um dos pontos. Seja específico.
    
    3. Uma conclusão
        - Escreva um parágrafo sobre conclusão.
    
    Podcast: {input}
    '''

prompt = PromptTemplate.from_template(template) # -> Forma rápida de escrever um prompt.

#Escolhi o modelo de IA que vou utilizar
chat = ChatGoogleGenerativeAI(
    model='models/gemini-flash-latest',
    google_api_key='SUA_CHAVE_AQUI'
)


 #Escolhi o modelo de IA para usar e defini o nível de criatividade da resposta
fluxo = prompt | chat # -> Especifiquei o fluxo de passos (montar o prompt -> IA responde)

#Criei uma função para dividir a transcrição em partes menores
def dividir_texto(texto, tamanho=1000):
    partes = []

    for indice in range(0, len(texto), tamanho):
        partes.append(texto[indice:indice+tamanho]) # -> append vai gardar cada pedaço na lista

    return partes


chunks = dividir_texto(final_transcricao)

respostas = []

for parcela in chunks:
    resposta = fluxo.invoke({'input': parcela})
    respostas.append(resposta.content)
    sleep(15)
    


print(respostas)





