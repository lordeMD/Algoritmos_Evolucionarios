import numpy as np
from cvrp_model import RouteDecoder

class GeneticAlgorithmCVRP:
    """
    Classe que implementa o Algoritmo Genético (AG) para resolver o
    Problema de Roteamento de Veículos Capacitado (CVRP).
    """
    def __init__(self, instance, pop_size=100, crossover_rate=0.8, 
                 mutation_rate=0.2, elitism_size=2, seed=42):
        """
        Inicializa o Algoritmo Genético.
        
        Args:
            instance (CVRPInstance): Instância do problema CVRP.
            pop_size (int): Tamanho da população.
            crossover_rate (float): Taxa de cruzamento (0 a 1).
            mutation_rate (float): Taxa de mutação (0 a 1).
            elitism_size (int): Número de melhores indivíduos a serem preservados diretamente.
            seed (int): Semente aleatória para reprodutibilidade.
        """
        self.instance = instance
        self.decoder = RouteDecoder(instance)
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_size = elitism_size
        
        # Semente do gerador aleatório do NumPy para garantir reprodutibilidade
        np.random.seed(seed)
        
    def initialize_population(self):
        """
        Inicializa a população de cromossomos aleatórios.
        Cada indivíduo é um Tour Gigante representado por uma permutação
        dos índices dos clientes (de 1 a N).
        
        Returns:
            list: Lista de listas, onde cada sublista é uma permutação dos clientes.
        """
        population = []
        customers = list(range(1, self.instance.num_customers + 1))
        
        for _ in range(self.pop_size):
            # Cria uma permutação aleatória dos clientes
            ind = list(np.random.permutation(customers))
            population.append(ind)
            
        return population
    
    def evaluate_population(self, population):
        """
        Avalia cada indivíduo da população calculando o custo total (distância).
        
        Args:
            population (list): População atual de cromossomos.
            
        Returns:
            tuple: (costs, decoded_routes)
                - costs (np.ndarray): Vetor contendo a distância total de cada cromossomo.
                - decoded_routes (list): Lista com as rotas decodificadas de cada indivíduo.
        """
        costs = np.zeros(self.pop_size)
        decoded_routes = []
        
        for idx, ind in enumerate(population):
            routes, dist = self.decoder.decode(ind)
            costs[idx] = dist
            decoded_routes.append(routes)
            
        return costs, decoded_routes


    def selection_tournament(self, population, costs, k=3):
        """
        Seleciona um indivíduo da população utilizando o método de Torneio Binário/K-ário.
        Escolhe K indivíduos aleatórios e retorna o que tiver o menor custo (menor distância).
        
        Args:
            population (list): População de cromossomos.
            costs (np.ndarray): Custos associados a cada indivíduo.
            k (int): Número de concorrentes no torneio (pressão seletiva).
            
        Returns:
            list: O cromossomo vencedor do torneio.
        """
        # Escolhe K índices aleatórios da população
        selected_indices = np.random.choice(self.pop_size, size=k, replace=False)
        
        # Identifica o índice do indivíduo com o menor custo (melhor aptidão)
        best_idx = selected_indices[np.argmin(costs[selected_indices])]
        
        # Retorna uma cópia do cromossomo selecionado
        return list(population[best_idx])


    def crossover_ox(self, parent1, parent2):
        """
        Aplica o operador de Cruzamento Ordenado (OX - Ordered Crossover).
        Específico para representações de permutação, pois preserva a ordem relativa
        dos elementos e evita duplicatas.
        
        Args:
            parent1 (list): Cromossomo do pai 1.
            parent2 (list): Cromossomo do pai 2.
            
        Returns:
            tuple: (child1, child2) Dois novos cromossomos filhos.
        """
        n = len(parent1)
        
        def generate_child(p1, p2):
            child = [-1] * n
            
            # 1. Seleciona dois pontos de corte aleatórios
            idx1 = np.random.randint(0, n - 1)
            idx2 = np.random.randint(idx1 + 1, n)
            
            # 2. Copia o segmento do p1 para o filho
            child[idx1:idx2] = p1[idx1:idx2]
            
            # 3. Preenche o restante do filho usando a ordem relativa do p2
            # Começamos a preencher logo após o segundo corte (ponto idx2)
            current_pos = idx2
            
            # Cicla sobre os elementos do p2 começando a partir de idx2
            p2_ordered = p2[idx2:] + p2[:idx2]
            
            for item in p2_ordered:
                if item not in child:
                    # Se alcançamos o final do vetor, voltamos ao início (circular)
                    if current_pos >= n:
                        current_pos = 0
                    child[current_pos] = item
                    current_pos += 1
            return child

        child1 = generate_child(parent1, parent2)
        child2 = generate_child(parent2, parent1)
        
        return child1, child2
