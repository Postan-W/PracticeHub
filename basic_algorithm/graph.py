#图的遍历
"""
从图G的顶点v出发，深度优先遍历（Depth First Search，DFS）的递归过程如下。
（1）访问顶点v，并为顶点v打上访问标记。
（2）选择顶点v未访问过的一个邻接顶点出发，深度优先遍历图G

宽度优先遍历（Breadth First Search，BFS）类似于树的层次遍历过程。从图G的顶点v出发，
宽度优先遍历的基本思想是访问顶点v并为顶点v打上访问标记。依次访问顶点v的所有未访问过的邻接点w1,w2,…,wk。依次从w1,w2,…,wk出发，依次访问它们的所有未访问过的邻接点。以此类推，直至图中所有与v有路径相通的顶点都被访问为止。
若G是连通图，则整个遍历结束；否则，需要另选一个未访问的顶点出发继续以上遍历过程。
"""
#最小生成树
"""
一个具有n个顶点的连通图的生成树是一个极小连通子图，它包括图中全部顶点，只有足以构成一棵树的n−1条边。对于一个连通图，
从不同的顶点出发进行遍历或采用不同的遍历方法，可以得到不同的生成树。如何在连通图众多的生成树中寻找一棵使得各条边上
的权值之和最小的生成树，是一个具有实际意义的问题。其中一个典型的应用是通信网络设计问题。假设要在n个城镇间建立通
信网络，至少要架设n−1条通信线路，每两个城市之间架设通信线路的成本并不一样，那么如何选择线路以使总造价最小呢？
一个带权连通图G的最小代价生成树是图G的所有生成树中，各边代价之和最小的生成树。
1.
普里姆（Prim）算法设G=(V,E)是带权连通图，T=(V',E')是正在构造中的生成树，选定从顶点v0开始构造
（也可以选择任意其他顶点），最小代价生成树的构造过程是：
（1）初始状态下，V'={v0}，E'={}，即生成树T中只有一个顶点v0，没有边。
（2）在所有u∈V'，v∈V−V'的边(u,v)∈E中找一条权值最小的边，将此边并入集合T中。
（3）重复步骤（2），直至V=V'。
2.
克鲁斯卡尔（Kruskal）算法设G=(V,E)是带权连通图，T=(V',E')是正在构造中的生成树，最小代价生成树的构造过程是：
（1）初始状态下，T=(V,{})，即生成树T中包含图G中的所有顶点，没有边。
（2）从E中选择代价最小的边(u,v)，若在T中加入边(u,v)后不会形成回路，则将其加入T中并从E中删除此边；否则，选择下一条代价最小的边。
（3）重复步骤（2），直至E'中包含n−1条边。
"""