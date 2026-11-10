# os/Path -> para lidar com caminhos de arquivos e variáveis de ambiente
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.skills import Skills, LocalSkills
from agno.tools.websearch import WebSearchTools
from agno.tools.file import FileTools # -> Ferramenta que permite o agente salvar/ler arquivos no disco




#2 - CONFIGURAÇÃO DE DIRETÓRIOS E FERRAMENTAS -------------------------
#    definimos onde salvar as skills e onde salvar os outputs



#Local onde está armazenado as skills (.md ou .txt)
skill_dir = Path(__file__).parent / "skills"



output_dir = Path(__file__).parent / "output/N1/" # -> Diretório

file_tools = FileTools(
    base_dir=output_dir, #diretório onde os arquivos serão salvos
    enable_save_file=True, #Permite ao FileTools salvar arquivos
    enable_read_file=True, #Permite ler arquivos existentes
    enable_list_files=True #Permite listar arquivos no diretório
    
)



# 3. CRIAÇÃO DO AGENTE
#    Aqui montamos o agente. A grande novidade é o parâmetro 'skills'.
#
#    • skills=Skills(...) → Instruímos o agente a carregar habilidades
#                           da pasta 'skills_dir'.
#
#    Isso permite que o código Python fique limpo e focado na lógica,
#    enquanto os prompts complexos ficam em arquivos de texto separados.
agente_noticias = Agent(
    name="Agente de Notícias",

    model=OpenAIResponses(id="gpt-5-nano", api_key=os.getenv("OPENAI_API_KEY")),

    # Skills: carrega habilidades da pasta local 'skills'
    # O agente vai ler os arquivos nessa pasta e entender como usar cada skill
    skills=Skills(loaders=[LocalSkills(str(skill_dir))]),




    # Instructions: define o comportamento, a persona e o pipeline do agente
    # Observe que agora referenciamos as skills pelos nomes dos arquivos/títulos
    instructions=[
        "Você é um jornalista completo: pesquisador, apurador, verificador factual e redator.",
        "Receba um tema e execute TODAS as etapas abaixo em sequência.",
        "",
        "Antes de começar, carregue as instruções de cada skill usando get_skill_instructions.",
        "",
        "ETAPA 1 — PESQUISA: use a skill 'pesquisa-noticias'",
        "ETAPA 2 — APURAÇÃO: use a skill 'apuracao-fontes'",
        "ETAPA 3 — VERIFICAÇÃO: use a skill 'verificacao-factual'",
        "ETAPA 4 — REDAÇÃO: use a skill 'redacao-jornalistica'",
        "",
        "Apresente ao usuário APENAS a matéria jornalística final (Etapa 4).",
        "As etapas 1, 2 e 3 são seu processo interno de trabalho.",
        "Salve o documento final em um arquivo .md no diretório",
    ],

    # Tools: ferramentas que o agente pode invocar durante a execução
    # - WebSearchTools → permite buscar informações na internet
    # - file_tools     → permite salvar a matéria final em .md
    tools=[WebSearchTools(), file_tools],

    # Injeta data/hora atual no contexto (útil para priorizar notícias recentes)
    add_datetime_to_context=True,

    # Formata a resposta do agente em Markdown
    markdown=True
)




if __name__ == "__main__":
    agente_noticias.print_response(
        "Trump questiona eleição brasileira e Alckmin rebate: “equivocado”"
    )








