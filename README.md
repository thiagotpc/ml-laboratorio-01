# ml-laboratorio-01
Laboratório 01 para disciplina de **Machine Learning** do PPGINF (2020/Período Especial)

## Objetivo
Neste laboratório, o objetivo era observar a qualidade de um classificador **knn** por meio, principalmente, de modificação nos parâmetros que definem o **vetor de características** para **arquivos de imagens em tons de cinza** que, aqui,  são representados pelos valores de *pixels* preto ou não preto que as compõem. Adicionalmente, pretendia-se verificar se era possível obter ainda melhor resultado ao se alterar valores para ***neighbors (k)*** e de **métricas de distância** do classificador.

### Enunciado da Atividade
A atividade esta assim descrita no Moodle da disciplina:
> Impactos da Representação
> O script digits.py extrai a represenação mais simples possível de uma base de dados dígitos manuscritos. Para cada posição da imagem, verifica-se o valor de intensidade do pixel e se esse valor for > 128, a característica é igual a 1, caso contrário 0. A imagens tem tamanho variável e como os classificadores precisam de um vetor de tamanho fixo, as imagens são normalizadas utilizando as variáveis X e Y dentro da função rawpixel. Após a execução do programa, um arquivo chamado features.txt é criado no diretório corrente. Esse arquivo contem 2000 linhas no formato
>
> 0 0:0 1:0 2:1 3:1
>
> O primeiro caractere indica o rótulo da classe. A sequencia i:v indica o índice da característica e o valor da mesma. Nesse caso, a características 0, 1, 2, e 3 tem valores 0, 0, 1 e 1, respectivamente.
>
> Sua tarefa consiste em gerar diferentes vetores de características variando os valores de X e Y. Utilizando um kNN (k=3 e distância Euclidiana), encontre o conjunto de características que produziu os piores e melhores resultados de classificação. A base de dados deve ser dividida em 50% para treinamento e 50% para validação. Compare as matrizes de confusão nesses dois casos e reporte quais foram as confusões resolvidas pela melhor representação.
>
> Para a sua melhor solução, verifique se é possível melhorar o resultados mudando os valores de k e métrica de distância.
>
> Escreva uma breve relatório reportando seus experimentos e entregue em formato PDF.
>
> IMPORTANTE:
>
> Somente arquivos em PDF serão corrigidos. 
> Respeite o prazo de entrega. O moodle não aceitará envios após a Data Limite.



## Estrutura de pastas e arquivos

- 📂digits
  - 📂data
    - *.jpg
  - files.txt
- 📂results
  - images-info.txt
  - hipoteses.csv
  - saidas_rodada_1.txt
  - saidas_rodada_2.txt
  - relatorio.pdf
- digits.py
- knn.py
- images-info.py
- rodada_1.bat
- rodada_2.bat

A pasta **digits/data** contém os arquivos da base de dados fornecida pelo professor, juntamente com o arquivo que rotula as imagens (**digits/files.txt**).

O professor também forneceu os scripts que geram o vetor de características (**digits.py**) e que realiza o treino e a classificação knn (**knn.py**). Estes foram modificados durante o laboratório.

Durante o laboratório foi criado um script para extrair algumas informações do conjunto de dados (**images-info.py**).

```python
# ...
def get_dimensoes_info_from_images():
    count_alturas_usadas = [0]*100
    count_larguras_usadas = [0]*100
    list_alturas = []
    list_larguras = []
    list_proporcao =[]

    folder_images = "digits/data"

    for dirpath, _, filenames in os.walk(folder_images):
        for path_image in filenames:
            image = os.path.abspath(os.path.join(dirpath, path_image))
            with Image.open(image) as img:
                width, heigth = img.size
                count_alturas_usadas[heigth] = count_alturas_usadas[heigth] + 1
                count_larguras_usadas[width] = count_larguras_usadas[width] + 1
                list_alturas.append(heigth)
                list_larguras.append(width)
                list_proporcao.append(width/heigth)

    print('LARGURA')
    print('min: ', min(list_larguras))
    print('max: ', max(list_larguras))
    print('media: ', statistics.mean(list_larguras))
    print('mediana: ', statistics.median(list_larguras))

    print('ALTURA')
    print('min: ', min(list_alturas))
    print('max: ', max(list_alturas))
    print('media: ', statistics.mean(list_alturas))
    print('mediana: ', statistics.median(list_alturas))

    print('PROPORCAO')
    print('media: ', statistics.mean(list_proporcao))
    print('mediana: ', statistics.median(list_proporcao))

    # histograma
    print('Histograma - Alturas')
    print(*count_alturas_usadas, sep='\n')
    
    print('Histograma - Larguras')
    print(*count_larguras_usadas, sep='\n')


if __name__ == "__main__":
    get_dimensoes_info_from_images()
```

Dois arquivos em lote para Windows foram criados. O primeiro (**rodada_1.bat**) para rodar as primeiras hipóteses de observação, mantendo k=3 e distância euclidiana para diferentes variações em dimensões na normalização do conjunto de dados. 

O segundo arquivo em lote (**rodada_2.bat**) permitia rodar todas as hipóteses de observação e gerar um único arquivo de saída (**results/rodada_2.bat**). O arquivo continha todas as variações propostas (hipóteses) para se observar melhores resultados, como exemplificado a seguir.

```bat
py knn.py features_10x10.txt 1 EU
py knn.py features_10x10.txt 1 MA
py knn.py features_10x10.txt 1 MI
py knn.py features_10x10.txt 3 EU
py knn.py features_10x10.txt 3 MA
py knn.py features_10x10.txt 3 MI
py knn.py features_10x10.txt 5 EU
py knn.py features_10x10.txt 5 MA
py knn.py features_10x10.txt 5 MI
...
py knn.py features_99x81.txt 1 EU
py knn.py features_99x81.txt 1 MA
py knn.py features_99x81.txt 1 MI
py knn.py features_99x81.txt 3 EU
py knn.py features_99x81.txt 3 MA
py knn.py features_99x81.txt 3 MI
py knn.py features_99x81.txt 5 EU
py knn.py features_99x81.txt 5 MA
py knn.py features_99x81.txt 5 MI
```

As saídas foram tabuladas manualmente (**results/hipotesis.csv**) e um relatório final foi redigido (**relatorio.pdf**) para entrega na atividade.

## Modificações nos scripts originais

### digits.py

Foi modificado para que pudesse ser gerado durante apenas uma execução todos os arquivos de vetores de características, de modo que fosse possível realizar algumas comparações entre eles, como, por exemplo, comparar seu tamanho.

A seguir exibimos parte do arquivo modificado. Outras adequações foram realizadas na função *main* para receber *n_neighbors* e *metric_distance* como parâmetros e pode ser conferido por *diff*.

```python
# ...
if __name__ == "__main__":

    # inicial e novas hipoteses, após analisar a base
	hipoteses = [
        {'largura': 10, 'altura': 10},
        {'largura': 10, 'altura': 30},
        {'largura': 12, 'altura': 15},
        {'largura': 20, 'altura': 10},
        {'largura': 20, 'altura': 25},
        {'largura': 20, 'altura': 70},
        {'largura': 30, 'altura': 40},
        {'largura': 32, 'altura': 40},
        {'largura': 36, 'altura': 46},
        {'largura': 37, 'altura': 47},
        {'largura': 40, 'altura': 50},
        {'largura': 50, 'altura': 60},
        {'largura': 60, 'altura': 30},
        {'largura': 60, 'altura': 70},
        {'largura': 99, 'altura': 81}
    ]

    for hipotese in hipoteses:
        largura = hipotese['largura']
        altura = hipotese['altura']
        filename = 'features_' + str(largura) + 'x' + str(altura) + '.txt'
        fout = open(filename, "w")
        images = load_images('digits/data', fout, largura, altura)
        fout.close
```

### knn.py

Foi modificado para receber como argumento de entrada valores para K e métricas de distância. Adicionalmente passou também a imprimir estas condições para que os resultados pudessem ser melhor identificados quando agrupados em uma única saída. 

A seguir exibimos parte do arquivo modificado. Outras adequações foram realizadas nas funções para receber largura e altura como parâmetros e pode ser conferido por *diff*.

```python
# ...
if __name__ == "__main__":

    avaiable_metrics = {'EU': 'euclidean', 'MA': 'manhattan', 'CH': 'chebyshev', 'MI': 'minkowski'}

    if len(sys.argv) < 1:
        sys.exit("Use: knn.py <data> [n_neighbors]")

    n_neighbors = 1
    if len(sys.argv) > 2:
        n_neighbors = int(sys.argv[2])

    metric_distance = 'euclidean'
    if len(sys.argv) > 3:
        metric_distance = avaiable_metrics[str(sys.argv[3])]

    main(sys.argv[1], n_neighbors, metric_distance)
```



## Comandos Utilizados

```cmd
# Para extrair informações do conjunto de imagens
py images-info.py > results\images-info.txt

# Para gerar os vetores de características em diferentes
py digits.py

# Para executar as hipótesis
rodada_1.bat > results/saidas_rodada_1.txt
rodada_2.bat > results/saidas_rodada_2.txt
```



## Ambiente

O laboratório aconteceu em computador notebook com Windows 10 e Python 3.8.5 (64bit).



## Limitações e Melhorias

Este projeto pode ser melhorado criando uma saída tabulada para se evitar a tabulação manual e outras informações podem ser calculados e registradas, como, por exemplo, o F1-Score e o tempo de execução dos scripts ou suas partes (normalização/vetorização/treinamento/classificação). Deste modo, pode-se obter uma nova interpretação para os parâmetros modificados.

