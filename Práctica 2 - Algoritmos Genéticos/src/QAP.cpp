
#include<iostream>
#include<fstream>
#include<algorithm>
#include<list>
#include<queue>

#include<chrono>
#include<sys/stat.h>
#include<dirent.h>
#include<ftw.h>
#include "../include/QAP.h"
#include "../include/random.h"
#include <random>
#include <vector>

using namespace std;
using namespace std::chrono;
typedef duration<float, ratio<1,1>> secondsf;


/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

int numIts = 0;

int nResultados = 0, num = 0;

static int P_C = 0, N_MUTATIONS = 0, FIN = 0, POBLATION = 0, SEED = 0;
static double P_M = 0.0;

/****************************************************************************/
/****************************************************************************/



/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

ostream& operator<<(ostream& os,  list<list<int> > l);

    QAP::QAP() {
        _n = 0;
        _dist = NULL;
        _w = NULL;
    }

    QAP::QAP(int n1) {
        _n = n1;
        _dist = new int* [_n];
        _w = new int* [_n];

        for(int i=0; i < _n; i++) {
            _dist[i] = new int[_n];
            _w[i] = new int[_n];
        }

        for(int i=0; i < _n; i++) {
            for(int j=0; j < _n; j++) {
                _dist[i][j] = 0;
                _w[i][j] = 0;
            }
        }
    }

    QAP::~QAP() {
        memoryFree();
    }

    void QAP::memoryFree() {
        for(int i=0; i < _n; i++) {
            delete [] _dist[i];
            delete [] _w[i];
        }

        delete [] _dist;
        delete [] _w;
        _n = 0;
        _dist = NULL;
        _w = NULL;
    }

    void QAP::saveMemory(const int t) {
        memoryFree();

        _dist = new int* [t];
        _w = new int* [t];

        for(int i=0; i < t; i++) {
            _dist[i] = new int [t];
            _w[i] = new int [t];
        }

        for(int i=0; i < t; i++) {
            for(int j=0; j < t; j++) {
                _dist[i][j] = 0;
                _w[i][j] = 0;
            }
        }

        _n = t;
    }


    int QAP::getElementDist(int i, int j) const{
        return _dist[i][j];
    }

    int QAP::getElementW(int i, int j) const{
        return _w[i][j];
    }

    int QAP::getN() const{
        return _n;
    }

    void QAP::setElementDist(int elemento, int i, int j) {
        _dist[i][j] = elemento;
    }

    void QAP::setElementW(int elemento, int i, int j) {
        _w[i][j] = elemento;
}

    void QAP::setN(int elemento) {
        _n = elemento;
    }

    QAP& QAP::operator= (const QAP &orig) {
        if(&orig != this) {
            saveMemory(orig.getN());
            this->_n = orig.getN();

            for(int i=0; i < _n; i++) {
                for(int j=0; j < _n; j++) {
                    this->_dist[i][j] = orig.getElementDist(i, j);
                    this->_w[i][j] = orig.getElementW(i, j);
                }
            }
        }
        return *this;
    }

/****************************************************************************/
/****************************************************************************/

    int QAP::distance(int i, list<int>&s, int k) {
        int d = 0, w, dist;
        list<int>::iterator its1, its2 = s.begin();

        its1 = its2 = s.begin();
        advance(its1, i);
        advance(its2, k);

        dist = this->_dist[i][k];
        w = this->_w[*its1][*its2];
        d += dist * w;

        return d;
    }

    int QAP::selectionCriteria(list<int>&s, list<int>&sel) {
        list<int>::iterator its, itsel;
        int iMax=0;
        int max = 0, d = 0;

        for(its = s.begin(); its != s.end(); ++its) {
            for(itsel = sel.begin(); itsel != sel.end(); ++itsel)
                d += distance(*itsel, s,*its);
            if(d > max) {
                max = d;
                iMax = *its;
            }
            d = 0;
        }

        return iMax;
    }

    list<int> QAP::Greedy(int firstBest) {
        list<int> sel, s;
        int k;
        list<int>::iterator itsel, its;
        for (int i=0; i < getN(); i++)
            s.push_back(i);

        k = firstBest;
        sel.push_back(k);
        s.remove(k);

        while(sel.size() != getN()) {
            k = selectionCriteria(s, sel);
            sel.push_back(k);
            s.remove(k);
        }

        return sel;
    }

/****************************************************************************/
/****************************************************************************/


int QAP::eval(list<int> s) {
    int dist, w, d = 0;
    list<int>::iterator its1, its2 = s.begin();

    for(int i=0; i < getN(); i++) {
        for(int j=0; j < getN(); j++) {
            its1 = its2 = s.begin();
            advance(its1, i);
            advance(its2, j);
            dist = this->_dist[i][j];
            w = this->_w[*its1][*its2];
            d += dist * w;
        }
    }

    return d;
}



    pair<list<int>, int> QAP::LS(list<int> individual, string type) {
        list<int> neighbour;
        int bestFitness;
        list<int>::iterator itsel, its;
        vector<int> shuffle;
        vector<int>::iterator it1, it2;
        for(int i=0; i < individual.size(); i++)
            shuffle.push_back(i);

        /*random_shuffle(shuffle.begin(), shuffle.end());
        it1 = it2 = shuffle.begin();
        ++it2;
        int k = Randint(0, getN());

        for(int i=0; i < k; i++) {
            neighbour = permute(individual, *it1, *it2);
            if (eval(neighbour) < eval(individual))
                individual = neighbour;
            ++it1;
            ++it2;
        }*/

        list<int> best = individual;
         for(int i=0; i < getN(); i++) {
             for(int j=i+1; j < getN(); j++) {
                 neighbour = permute(individual, i, j);
                 if(eval(neighbour) < eval(individual)) {
                     cout << endl << eval(neighbour);
                     if(type == "Lamarck")
                         individual = neighbour;

                     bestFitness = eval(neighbour);
                 }
             }
         }
         /*
         list<int> best = individual;
                  for(int i=0; i < getN(); i++) {
                      for(int j=i+1; j < getN(); j++) {
                          neighbour = permute(individual, i, j);
                          if (eval(neighbour) < eval(individual)) {
                              cout << endl << eval(neighbour);
                              individual = neighbour;
                          }
                      }

        for(int i=0; i < individual.size(); i++)
            shuffle.push_back(i);

        int N_ITS = 60;
        list<int> best = individual;
        for(int j=0; j < N_ITS; j++)  {
            //int k = Randint(0, getN());
            random_shuffle(shuffle.begin(), shuffle.end());
            it1 = it2 = shuffle.begin();
            ++it2;
            best = individual;
            for(int i=0; i < N_ITS; i++) {
                neighbour = permute(individual, *it1, *it2);
                if(eval(neighbour) < eval(best))
                    best = neighbour;
                ++it1; ++it2;
            }
            if(eval(best) < eval(individual)) {
                if(type == "Lamarck")
                    individual = best;

                bestFitness = eval(individual);
            }
        }*/
         /*

            *********Local************
            for(int i=0; i < individual.size(); i++)
                shuffle.push_back(i);

            do {
                random_shuffle(shuffle.begin(), shuffle.end());
                it1 = it2 = shuffle.begin();
                ++it2;
                do {
                    neighbour = permute(individual, *it1, *it2);
                    ++it1; ++it2;
                } while(eval(neighbour) > eval(individual));

                if(eval(neighbour) < eval(individual))
                     individual = neighbour;
            } while(eval(neighbour) < eval(individual));



            for(int i=0; i < individual.size(); i++) {
                for(int j=i+1; j < individual.size(); j++) {
                    neighbour = permute(individual, i, j);
                    if(eval(neighbour) < eval(individual))
                         individual = neighbour;
                 }
             }
          */
          pair<list<int>, int> par;
          par.first = individual;
          par.second = bestFitness;
          return par;
    }
    /**************************************************************************/
    /**************************************************************************/
    /**************************************************************************/
    /**************************************************************************/
    /**************************************************************************/
    /**************************************************************************/
    /**************************************************************************/


    list<int> QAP::permute(list<int> individual, int i, int j) {
        int aux;
        list<int>::iterator it, it2;
        list<int> individualAux = individual;
        it = it2 = individualAux.begin();
        advance(it, i);
        advance(it2, j);

        aux = *it;
        *it = *it2;
        *it2 = aux;
        return individualAux;
    }

    list<int> QAP::best(list<int> &n1, list<int> &n2) {
        list<int> p;

        if(eval(n1) < eval(n2))
            p = n1;

        else
            p = n2;

        return p;
    }

    //Mediante torneo binario: se eligen aleatoriamente 2 individuos y se
    //selecciona el mejor de ellos
    // En el GG se aplica el torneo tantas veces como individuos hayan
    list<list<int> > QAP::select(priority_queue<pair<list<int>, int>,
    vector<pair<list<int>, int> >, OrdenGenetico> poblacion) {
        list<int> element, element2;
        vector<int> shuffle;
        vector<int>::iterator itSuffle;
        list<list<int> > elements;
        list<list<int>>::iterator it = elements.begin();
        list<list<int>> individuals;
        
        priority_queue<pair<list<int>, int>,
        vector<pair<list<int>, int> >, OrdenGenetico> aux = poblacion;

        individuals.push_back(aux.top().first);
        while(!aux.empty()) {
            elements.push_back(aux.top().first);
            aux.pop();
        }

        for(int i=0; i < elements.size(); i++)
            shuffle.push_back(i);

        random_shuffle(shuffle.begin(), shuffle.end());
        itSuffle = shuffle.begin();

        it = elements.begin();

        advance(it, *itSuffle);
        element = *it;
        ++itSuffle;

        it = elements.begin();

        advance(it, *itSuffle);
        element2 = *it;
        ++itSuffle;

        individuals.push_back(best(element, element2));
        //cout << "Father and Mother: ";
        list<int> e = best(element, element2);
        //cout << e << endl;

        return individuals;
    }

    //Los hijos sustituyen a la poblacion total
    // En el elitista, los hijos generados en el cruce y mutacion  sustituyen
    // a los 2 peores de la poblacion actual
    void QAP::replace(list<int> &child, priority_queue<pair<list<int>, int>, vector<pair<list<int>, int> >,
    OrdenGenetico> &poblation, int poblationSize) {
        priority_queue<pair<list<int>, int>, vector<pair<list<int>, int> >,
        OrdenGenetico> aux;
        pair<list<int>, int> par, element;

	    par.first = child;
	    par.second = eval(child);

        for(int i = 0; i < poblationSize-1; i++) {
            element.first = poblation.top().first;
            element.second = poblation.top().second;
            aux.push(element);
            poblation.pop();
        }
        aux.push(par);
        poblation = aux;
    }

    // Cruce = Recombinar
    // Los hijos heredan caracteristicas de cada padre, si no es asi entonces
    // esto sera un operador de mutacion
    // Los valores que tengan la misma posicion en los padres se mantienen en los hijos
    // 2 tipos de cruces:
    // Cruce uniforme: Genera un hijo a partir de 2 padres, el resto de posiciones
    // se autocompletan con los valores de un padre u otro, necesita reparador
    // Cruce basado en posicion: El resto de posiciones se cogen de un padre, da igual
    // de cual, se crean ordenes aleatorios y se completa el hijo, mas dificil que converja
    list<int> QAP::cross(list<list<int>> parents) {
        list<int> father, mother, child;
        list<list<int> >::iterator it;
        list<int>::iterator iter, itList, ini;

        for(int i=0; i < 2; i++) {
            it = parents.begin();
            father = *it;
            advance(it, 1);
            mother = *it;
        }

        ini = father.begin();
        advance(ini, getN()/2);

        for(itList = ini; itList != father.end(); ++itList)
            child.push_back(*itList);

        itList = mother.begin();
        while(child.size() != getN()) {
            if(find(child.begin(), child.end(), *itList) == child.end())
                child.push_back(*itList);
            else
                ++itList;
        }

        return child;
    }

    // Probabilidad de mutar muy baja
    void QAP::mutate(list<int> &child, int nMutations) {
        int probMutate = P_M * getN();
        int element = Randint(0, getN()), a, b, aux;

        if(element < probMutate) {
            list<int>::iterator it, it2;
            vector<int> shuffle;
            vector<int>::iterator itShuffle;

            for(int i=0; i < getN(); i++)
                shuffle.push_back(i);

            random_shuffle (shuffle.begin(), shuffle.end());

            itShuffle = shuffle.begin();
            for(int i=0; i < nMutations; i++) {
                it = it2 = child.begin();
                a = *itShuffle;
                ++itShuffle;
                b = *itShuffle;

                advance(it, a);
                advance(it2, b);

                aux = *it;
                *it = *it2;
                *it2 = aux;
            }
        }

    }


    list<list<int>> QAP::initialize(int poblationSize) {
        list<int> s;
        list<list<int>> initial;
        vector<int> shuffle;
        vector<int>::iterator itShuffle;

        for(int i=0; i < getN(); i++)
            shuffle.push_back(i);

        for(int i=0; i < poblationSize; i++) {
            s.clear();
            random_shuffle (shuffle.begin(), shuffle.end());

            for(itShuffle = shuffle.begin(); itShuffle != shuffle.end(); ++itShuffle)
                s.push_back(*itShuffle);
            initial.push_back(s);
        }

        return initial;
    }

    //Generacional: En cada iteracion se crea una nueva poblacion que reemplazar
    //completamente a la anterior
    //Estacionaria: En cada iteracion se eligen 2 padres y se le aplica los
    //operadores geneticos, reemplazando a 2 individuos de la poblacion anterior
    // produciendo presion selectiva alta (convergencia rapida) porque
    //reemplaza a los peores cromosomas de la poblacion


    pair<list<int>, int> QAP::GeneticAlgorithm (int fin, int poblacionSize, string type) {
        priority_queue<pair<list<int>, int>, vector<pair<list<int>, int> >, OrdenGenetico> poblation, aux;
        pair<list<int>, int> values;
        list<list<int>> poblationInitial = initialize(poblacionSize);
        
        list<list<int>>::iterator itIndividuals;
        
        for(itIndividuals = poblationInitial.begin(); itIndividuals != poblationInitial.end(); ++itIndividuals) {
            values.first = *itIndividuals;
            values.second = eval(*itIndividuals);
            poblation.push(values);
            //cout << values.first << " Eval: " << values.second << endl;
        }

        list <int> best = poblation.top().first;
        //cout << "Best: " << best << " Eval: " << poblation.top().second << endl;
        //cout << endl << "-------------------------------------------------------------";

        for(int i=0; i < fin; i++) {
            best = poblation.top().first;
            best = LS(best, type);
            values.first = best;
            values.second = eval(best);
            poblation.pop();
            poblation.push(values);
            cout << endl << "i: " << i << " - " << poblation.top().second;
            //cout << endl << "-------------------------------------------------------------" << endl;
            list<list<int>> parents = select(poblation);

            list<int> child = cross(parents);
            //cout << "child: " << child << " Eval: " << eval(child) << endl;

            mutate(child, N_MUTATIONS);

            replace(child, poblation, poblacionSize);

            best = poblation.top().first;
            list<int>::iterator firstBest = best.begin();
            aux = poblation;

            /*while(!poblation.empty()) {
                best = poblation.top().first;
                LS(best);
                values.first = best;
                values.second = eval(best);
                aux.push(values);
                poblation.pop();
            }
            poblation = aux;

            while(!aux.empty())
                aux.pop();*/
        }
        
        values = poblation.top();

        return values;
    }
    /**
	 * @brief Metodo para escribir por flujo de salida objetos de la clase QAP
	 * @param os Flujo de salida
	 * @param MD Objeto de la clase QAP
	 * @return Devuelve el flujo de salida
	*/
ostream& operator<<(ostream& os, const QAP& qap) {
    for (int i=0; i < qap.getN(); i++) {
        for (int j=0; j < qap.getN(); j++)
            os << endl << "Para i=" << i << ", j=" << j << ", dist="
            << qap.getElementDist(i, j) << " y w=" << qap.getElementW(i, j);
    }

    return os;
}

ostream& operator<<(ostream& os,  list<list<int> > l) {
  list<list<int> >::iterator its;
  list<int>::iterator itsel;

  for(its = l.begin(); its != l.end(); ++its) {
    for(itsel =its->begin(); itsel != its->end(); ++itsel)
      cout << " " << *itsel;
    cout << endl;
  }

    return os;
}


    /**
	 * @brief Metodo para leer por flujo de entrada objetos de la clase QAP
	 * @param is Flujo de entrada
	 * @param MD Objeto de la clase QAP
	 * @return Devuelve el flujo de entrada
	*/
istream& operator>>(istream& is, QAP& MD) {

    int n, i = 0, j = 0, dist, c=0;

    is >> n;

    QAP qap(n);
    qap.setN(n);

    while(is >> dist) {
        if (c < qap.getN() * qap.getN())
            qap.setElementDist(dist, i, j);
        else if (c == qap.getN() * qap.getN()) {
            j = 0;
            i = 0;
        }
        else
            qap.setElementW(dist, i, j);
        j++;
        c++;
        j %= n;
        if(j == 0)
            i++;
    }

    MD = qap;
    return is;
}


void save2File(ofstream fout, pair<list<int>, int> best, int seconds, string type) {

    fout << "-----------------------------------------------------" << endl;
    fout << "-----------------------------------------------------" << endl;

    fout << "Eval: " << best.second << " - seconds: " << seconds
    << " - type: " << type << " - P_M: " << P_M << " - N_MUTATIONS: " << N_MUTATIONS
    << " - Poblation: " << POBLATION << " - FIN: " << FIN << " - Inidividual: " << best.first << endl;
    fout << "-----------------------------------------------------" << endl;
    fout << "-----------------------------------------------------" << endl;

}


int main(int argc, char* argv[]) {

    if (argc < 2) {
        cerr << endl << endl << "Error: Funcionamiento del programa: ./MD (archivo)";
        exit(-1);
    }

    ifstream f;
    ofstream fout;

    f.open(argv[1]);
    if(!f.is_open()) {
        cerr << endl << endl << "Error: No se pudo abrir el archivo";
        exit(-1);
    }
    
    fout.open("../salida/res.txt", ios::app);
    if(!fout.is_open()) {
        cerr << endl << endl << "Error: No se pudo abrir el archivo de salida";
        exit(-1);
    }

    QAP qap;
    f >> qap;

    /*list<int> poblations, mutations, fin;
    poblations.push_back(10);
    poblations.push_back(20);
    poblations.push_back(40);
    poblations.push_back(80);
    poblations.push_back(100);
    mutations.push_back(1);
    mutations.push_back(2);
    mutations.push_back(4);
    mutations.push_back(8);
    mutations.push_back(10);
    fin.push_back(40);
    fin.push_back(20);
    fin.push_back(15);
    fin.push_back(10);
    fin.push_back(4);*/

    P_M = 0.1, N_MUTATIONS = 10, FIN = 20, POBLATION = 10, SEED = 20;
    //srand(SEED);

    time_point<steady_clock> ini = steady_clock::now();
    pair<list<int>, int> best = qap.GeneticAlgorithm(FIN, POBLATION, "Lamarck");
    time_point<steady_clock> fin = steady_clock::now();
    steady_clock::duration duration = fin-ini;

    cout << endl << endl << "Best: " << best.second
    << " - seconds: " << secondsf(duration).count() << " - (seconds per iteration): " << secondsf(duration).count()/FIN;

    fout.close();
    f.close();
}
