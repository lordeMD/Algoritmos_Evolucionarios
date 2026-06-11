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