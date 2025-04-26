
#Importando a biblioteca gradio
import gradio as gr
#importando a biblioteca do huggingface
from transformers import pipeline

# Carregando o modelo Summarization, que resume um texto
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Função que o Gradio vai usar, o texto dado pelo usuario sera alocado nessa função
def gerar_resumo(texto):
    #condições do texto: não pode estar vazio e não pode ter menos do que 30 caractere
    if not texto or len(texto.strip()) < 30:
        return "Insira um texto com pelo menos 30 caracteres."
    
    # O texto só podera ter até 3000 tokens
    texto = texto[:3000]
    
    #caso tudo esteja correto
    try:
        #O texto vai ser passado para o modelo de resumir
        resumo = summarizer(texto, max_length=200, min_length=30, do_sample=False)
        return resumo[0]['summary_text']   # <-- lista de dicionario
    except Exception as e:
        #caso tenha erros, não acontecerá nada
        return f"Erro ao gerar resumo: {e}"

# Criando a interface da pagina, usando o gradio
interface = gr.Interface(
    fn=gerar_resumo,
    #onde será inserido o texto
    inputs=gr.Textbox(lines=15, placeholder="Cole aqui seu conteúdo de estudo...", label="Texto de Estudo"),
    #onde sairá o resumo
    outputs=gr.Textbox(lines=15, label="Resumo Gerado"),
    title="Assistente de Estudo com IA"
)

# Executa o app
if __name__ == "__main__":
    interface.launch()
