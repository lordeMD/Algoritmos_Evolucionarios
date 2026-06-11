import matplotlib.pyplot as plt
import numpy as np

# Configuração global de estilo para gráficos elegantes e modernos
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#D3D3D3'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#F0F0F0'
plt.rcParams['grid.linestyle'] = '--'

def plot_routes(instance, routes, save_path=None):
    """
    Plota as rotas decodificadas em um gráfico bidimensional.
    Exibe o depósito central, a localização de cada cliente com sua demanda,
    e as rotas de cada veículo diferenciadas por cores harmônicas.
    
    Args:
        instance (CVRPInstance): Instância do problema.
        routes (list of lists): Rotas geradas (cada rota é uma lista de IDs de clientes).
        save_path (str, opcional): Caminho do arquivo para salvar a imagem (ex: 'rotas.png').
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.set_facecolor('#FAFAFA')
    
    # 1. Definir paleta de cores elegantes para as rotas (evitando cores primárias puras)
    colors = [
        '#1E3A8A', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
        '#EC4899', '#06B6D4', '#64748B', '#F97316', '#14B8A6',
        '#84CC16', '#A855F7', '#6366F1', '#D946EF', '#D97706'
    ]
    
    # 2. Plotar os Clientes
    coords = instance.coords
    demands = instance.demands
    
    # Clientes (excluindo o depósito 0)
    ax.scatter(coords[1:, 0], coords[1:, 1], color='#374151', s=60, 
               zorder=4, label='Clientes', edgecolors='#1F2937', linewidths=0.5)
    
    # Adicionar IDs e demandas nos clientes
    for i in range(1, len(coords)):
        ax.text(coords[i, 0] + 1.2, coords[i, 1] - 0.5, f"C{i}\n({int(demands[i])})", 
                fontsize=8, color='#4B5563', weight='bold',
                verticalalignment='center', horizontalalignment='left')

    # 3. Plotar o Depósito Central (0) com destaque
    ax.scatter(coords[0, 0], coords[0, 1], color='#DC2626', s=160, marker='s',
               zorder=5, label='Depósito', edgecolors='#991B1B', linewidths=1.5)
    ax.text(coords[0, 0] + 1.8, coords[0, 1] + 1.8, "DEPÓSITO", 
            fontsize=10, color='#991B1B', weight='bold',
            verticalalignment='bottom', horizontalalignment='left', zorder=6)

    # 4. Plotar as Rotas
    for route_idx, route in enumerate(routes):
        color = colors[route_idx % len(colors)]
        
        # Reconstrói a rota completa começando e terminando no depósito (0)
        full_route = [0] + route + [0]
        
        # Coleta as coordenadas x e y da rota
        x_coords = [coords[node, 0] for node in full_route]
        y_coords = [coords[node, 1] for node in full_route]
        
        # Desenha a linha da rota com um tom suave e espessura agradável
        ax.plot(x_coords, y_coords, color=color, linewidth=2.0, alpha=0.85, 
                zorder=3, label=f"Veículo {route_idx + 1} (Q={sum(demands[c] for c in route):.0f})")
        
        # Adiciona setas direcionais nas arestas para indicar o sentido da rota
        for i in range(len(full_route) - 1):
            start_x, start_y = coords[full_route[i]]
            end_x, end_y = coords[full_route[i+1]]
            
            # Calcula vetor diretor e ponto médio para plotar a seta no meio do caminho (evita poluição visual)
            mid_x = (start_x + end_x) / 2.0
            mid_y = (start_y + end_y) / 2.0
            dx = end_x - start_x
            dy = end_y - start_y
            
            # Normaliza o vetor para tamanhos consistentes das setas
            length = np.hypot(dx, dy)
            if length > 0:
                dx_norm = dx / length * 1.5
                dy_norm = dy / length * 1.5
                # Desenha uma ponta de seta elegante
                ax.annotate("", xy=(mid_x + dx_norm, mid_y + dy_norm), xytext=(mid_x - dx_norm, mid_y - dy_norm),
                            arrowprops=dict(arrowstyle="-|>", color=color, mutation_scale=12, lw=0, fill=True),
                            zorder=3)

    # 5. Embelezamento do Gráfico
    ax.set_title("Rotas Otimizadas do CVRP via Algoritmo Genético", fontsize=14, weight='bold', pad=15, color='#1F2937')
    ax.set_xlabel("Coordenada X (km)", fontsize=11, labelpad=8, color='#4B5563')
    ax.set_ylabel("Coordenada Y (km)", fontsize=11, labelpad=8, color='#4B5563')
    
    # Ocultar as bordas de cima e da direita para design minimalista (estilo seaborn/ggplot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, zorder=1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Gráfico de rotas salvo em: {save_path}")
    
    plt.show()

def plot_convergence(history_best, history_avg, save_path=None):
    """
    Plota as curvas de convergência do algoritmo ao longo das gerações.
    Mostra o comportamento do melhor custo e do custo médio da população.
    
    Args:
        history_best (list): Histórico do menor custo (distância) por geração.
        history_avg (list): Histórico do custo médio por geração.
        save_path (str, opcional): Caminho do arquivo para salvar a imagem.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.set_facecolor('#FAFAFA')
    
    generations = range(1, len(history_best) + 1)
    
    # Plotar Melhor Custo com linha sólida azul marinho
    ax.plot(generations, history_best, color='#1E3A8A', linewidth=2.0, label='Melhor Custo (Elitismo)')
    
    # Plotar Custo Médio com linha tracejada laranja
    ax.plot(generations, history_avg, color='#F59E0B', linewidth=1.5, linestyle='--', label='Custo Médio da População')
    
    # Ponto de ótimo encontrado (fim da curva de melhor custo)
    best_gen = np.argmin(history_best)
    best_cost = history_best[best_gen]
    ax.scatter(best_gen + 1, best_cost, color='#DC2626', s=50, zorder=5)
    ax.annotate(f"Melhor: {best_cost:.2f}", xy=(best_gen + 1, best_cost), 
                xytext=(best_gen + 1 + len(generations)*0.02, best_cost + (max(history_avg)-min(history_best))*0.05),
                arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.0),
                fontsize=9, color='#DC2626', weight='bold')

    # Embelezamento do Gráfico
    ax.set_title("Curva de Convergência do Algoritmo Genético", fontsize=14, weight='bold', pad=15, color='#1F2937')
    ax.set_xlabel("Gerações", fontsize=11, labelpad=8, color='#4B5563')
    ax.set_ylabel("Custo Total (Distância)", fontsize=11, labelpad=8, color='#4B5563')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, zorder=1)
    ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Gráfico de convergência salvo em: {save_path}")
        
    plt.show()
