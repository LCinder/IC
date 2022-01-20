

#ifndef MDP_H
#define MDP_H


#include<iostream>
#include<fstream>
#include<algorithm>
#include<list>
#include<queue>

#include<chrono>
#include<sys/stat.h>
#include<dirent.h>
#include<ftw.h>
#include "random.h"

using namespace std;
using namespace std::chrono;
typedef duration<float, ratio<1,1>> secondsf;


/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

static const int NUM_MAX_ITS = 400;

/****************************************************************************/
/****************************************************************************/

struct Orden {
    bool operator()(const pair<int, int> &n, const pair<int, int> &n1) {
        return n.second > n1.second;
    }
};

struct OrdenGenetico {
		bool operator()(const pair<list<int>, int> &n, const pair<list<int>, int> &n1) {
				return n.second > n1.second;
		}
};

struct OrdenEnteros {
		bool operator()(const pair<int, int> &n, const pair<int, int> &n1) {
				return n.second < n1.second;
		}
};


ostream& operator<<(ostream& os, const priority_queue<pair<int, int>, vector<pair<int, int> >, Orden> &p) {
    priority_queue<pair<int, int>, vector<pair<int, int> >, Orden> p_ = p;
    while(!p_.empty()) {
        cout << " (" << p_.top().first << " - " << p_.top().second<< ") ";
        p_.pop();
    }

    return os;
}


ostream& operator<<(ostream& os, const priority_queue<pair<int, int>, vector<pair<int, int> > > &p) {
    priority_queue<pair<int, int>, vector<pair<int, int> > > p_ = p;
    while(!p_.empty()) {
        cout << " (" << p_.top().first<< " - " << p_.top().second<< ") ";
        p_.pop();
    }

    return os;
}

ostream& operator<<(ostream& os,  list<int> &sel) {
        list<int>::iterator itsel;
    for(itsel = sel.begin(); itsel != sel.end(); ++itsel)
        os << " " << *itsel;

    return os;
}


/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

class QAP {

private:
/**
 * @page Maxima Diversidad
 * @author Diego Hernandez Moreno
 * @section invPosicion Invariante de Representacion de QAP
 * El invariante es \e size > 0
 * * @section faPosicion Funcion de Abstraccion de QAP
 * Un objeto valido @e rep del TDA QAP es
 * (rep._n, rep._m)
 */
    int _n;/**< Numero de elementos totales*/
    int** _dist;/**< MAtriz que guarda distancias entre elementos*/
    int** _w;/**< MAtriz que guarda distancias entre elementos*/

public:
    /**
	 * @brief Constructor de QAP
	*/
    QAP();
    /**
	 * @brief Constructor de QAP con parametros
	 * @param n1 Numero elementos totales
	 * @param m1 Numero elementos a elegir
	*/
    QAP(int n1);
    /**
	 * @brief Destructor de QAP
	*/
    ~QAP();
    /**
	 * @brief Libera memoria dinamica
	*/
    void memoryFree();
    /**
	 * @brief Reserva memoria dinamica para un numero de elementos pasado por argumento
	 * @param t Numero de elementos totales nuevo, guardando todo lo anterior
	*/
    void saveMemory(const int t);
    /**
	 * @brief Devuelve el elemento de la posicion en la matriz i y j
	 * @param i Elemento en la fila i
	 * @param j Elemento en la columna j
	 * @return Distancia entre los elementos i y j
	*/
    int getElementDist(int i, int j) const;
    int getElementW(int i, int j) const;

    /**
	 * @brief Devuelve el numero de elementos totales
	 * @return Elementos totales
	*/
    int getN() const;
    /**
	 * @brief Cambia el elemento de la posicion i y j por el pasado por argumento
	 * @param elemento Elemento a cambiar en las posiciones dadas
	 * @param i Fila i
	 * @param j Columna j
	*/
    void setElementDist(int elemento, int i, int j);
    void setElementW(int elemento, int i, int j);
    /**
	 * @brief Cambia el numero de elementos totales
	 * @param elemento N a cambiar
	*/
    void setN(int elemento);
    /**
	 * @brief Operador de copia que asigna un objeto de la clase a otro
	 * @param orig Elemento de la clase del que se van a asignar los datos
	 * @return Elemento de la clase *this con los parametros cambiados
	*/
    QAP& operator= (const QAP &orig);

/****************************************************************************/
/****************************************************************************/
    /**
	 * @brief Metodo que devuelve la distancia entre dos elementos
	 * @param i Elemento primero
	 * @param s Lista que contiene esos dos elementos
	 * @param k Elemento segundo
	 * @return Distancia entre dos elementos
	*/
    int distance(int i, list<int>&s, int k);
    
    list<int> permute(list<int> individual, int i, int j);
    
    pair<list<int>, int> LS (list<int> individual, stirng type);
    /**
	 * @brief Metodo que decide que elemento escoger sabiedo que elige el elemento mas lejano entre la lista de no seleccionados con respecto a la lista de seleccionados
	 * @param s Lista de elementos no seleccionados
	 * @param sel Lista de elementos seleccionados
	 * @return Indice del elemento mas lejano
	*/
    int selectionCriteria(list<int>&s, list<int>&sel);
    /**
	 * @brief Algoritmo Voraz que devuelve una lista de elementos seleccionados que aportan la maxima diversidad
	 * @return Lista de elementos cuya distancia es maxima
	*/
    list<int> Greedy(int firstBest);

/****************************************************************************/
/****************************************************************************/

    /**
	 * @brief Metodo que devuelve la distancia entre dos elementos
	 * @param n Elemento primero
	 * @param k Elemento segundo
	 * @return Distancia entre los dos elementos
	*/
    int eval(list<int> s);
    
    list<list<int>> initialize(int poblationSize);
    
    list<list<int> > select(priority_queue<pair<list<int>, int>,
    vector<pair<list<int>, int> >, OrdenGenetico> poblacion);

    pair<list<int>, int> GeneticAlgorithm(int fin, int poblacionSize, string type);
    
    list<int> best(list<int> &n1, list<int> &n2);

    void replace(list<int> &child, priority_queue<pair<list<int>, int>, vector<pair<list<int>, int> >,
    OrdenGenetico> &poblation, int poblationSize);

    list<int> cross(list<list<int> > parents);
    
    void mutate(list<int> &child, int nMutations);
    
    friend ostream& operator<<(ostream& os, const QAP& mdiversidad);
    
    friend istream& operator>>(istream& is, QAP& MD);
};



#endif /* MDP_H */
