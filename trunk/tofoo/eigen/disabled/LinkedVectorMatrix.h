// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_LINKEDVECTORMATRIX_H
#define EIGEN_LINKEDVECTORMATRIX_H

template<typename _Scalar, int _Flags>
struct ei_traits<LinkedVectorMatrix<_Scalar,_Flags> >
{
  typedef _Scalar Scalar;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = SparseBit | _Flags,
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    SupportedAccessPatterns = InnerCoherentAccessPattern
  };
};

template<typename Element, int ChunkSize = 8>
struct LinkedVectorChunk
{
  LinkedVectorChunk() : next(0), prev(0), size(0) {}
  Element data[ChunkSize];
  LinkedVectorChunk* next;
  LinkedVectorChunk* prev;
  int size;
  bool isFull() const { return size==ChunkSize; }
};

template<typename _Scalar, int _Flags>
class LinkedVectorMatrix
  : public SparseMatrixBase<LinkedVectorMatrix<_Scalar,_Flags> >
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(LinkedVectorMatrix)
    class InnerIterator;
  protected:

    enum {
      RowMajor = Flags&RowMajorBit ? 1 : 0
    };

    struct ValueIndex
    {
      ValueIndex() : value(0), index(0) {}
      ValueIndex(Scalar v, int i) : value(v), index(i) {}
      Scalar value;
      int index;
    };
    typedef LinkedVectorChunk<ValueIndex,8> VectorChunk;

    inline int find(VectorChunk** _el, int id)
    {
      VectorChunk* el = *_el;
      while (el && el->data[el->size-1].index<id)
        el = el->next;
      *_el = el;
      if (el)
      {
        // binary search
        int maxI = el->size-1;
        int minI = 0;
        int i = el->size/2;
        const ValueIndex* data = el->data;
        while (data[i].index!=id)
        {
          if (data[i].index<id)
          {
            minI = i+1;
            i = (maxI + minI)+2;
          }
          else
          {
            maxI = i-1;
            i = (maxI + minI)+2;
          }
          if (minI>=maxI)
            return -1;
        }
        if (data[i].index==id)
          return i;
      }
      return -1;
    }

  public:
    inline int rows() const { return RowMajor ? m_data.size() : m_innerSize; }
    inline int cols() const { return RowMajor ? m_innerSize : m_data.size(); }

    inline const Scalar& coeff(int row, int col) const
    {
      const int outer = RowMajor ? row : col;
      const int inner = RowMajor ? col : row;

      VectorChunk* el = m_data[outer];
      int id = find(&el, inner);
      if (id<0)
        return Scalar(0);
      return el->data[id].value;
    }

    inline Scalar& coeffRef(int row, int col)
    {
      const int outer = RowMajor ? row : col;
      const int inner = RowMajor ? col : row;

      VectorChunk* el = m_data[outer];
      int id = find(&el, inner);
      ei_assert(id>=0);
//       if (id<0)
//         return Scalar(0);
      return el->data[id].value;
    }

  public:

    inline void startFill(int reserveSize = 1000)
    {
      clear();
      for (unsigned int i=0; i<m_data.size(); ++i)
        m_ends[i] = m_data[i] = 0;
    }

    inline Scalar& fill(int row, int col)
    {
      const int outer = RowMajor ? row : col;
      const int inner = RowMajor ? col : row;
//       std::cout << " ll fill " << outer << "," << inner << "\n";
      if (m_ends[outer]==0)
      {
        m_data[outer] = m_ends[outer] = new VectorChunk();
      }
      else
      {
        ei_assert(m_ends[outer]->data[m_ends[outer]->size-1].index < inner);
        if (m_ends[outer]->isFull())
        {

          VectorChunk* el = new VectorChunk();
          m_ends[outer]->next = el;
          el->prev = m_ends[outer];
          m_ends[outer] = el;
        }
      }
      m_ends[outer]->data[m_ends[outer]->size].index = inner;
      return m_ends[outer]->data[m_ends[outer]->size++].value;
    }

    inline void endFill() { }

    void printDbg()
    {
      for (int j=0; j<m_data.size(); ++j)
      {
        VectorChunk* el = m_data[j];
        while (el)
        {
          for (int i=0; i<el->size; ++i)
            std::cout << j << "," << el->data[i].index << " = " << el->data[i].value << "\n";
          el = el->next;
        }
      }
      for (int j=0; j<m_data.size(); ++j)
      {
        InnerIterator it(*this,j);
        while (it)
        {
          std::cout << j << "," << it.index() << " = " << it.value() << "\n";
          ++it;
        }
      }
    }

    ~LinkedVectorMatrix()
    {
      clear();
    }

    void clear()
    {
      for (unsigned int i=0; i<m_data.size(); ++i)
      {
        VectorChunk* el = m_data[i];
        while (el)
        {
          VectorChunk* tmp = el;
          el = el->next;
          delete tmp;
        }
      }
    }

    void resize(int rows, int cols)
    {
      const int outers = RowMajor ? rows : cols;
      const int inners = RowMajor ? cols : rows;

      if (this->outerSize() != outers)
      {
        clear();
        m_data.resize(outers);
        m_ends.resize(outers);
        for (unsigned int i=0; i<m_data.size(); ++i)
          m_ends[i] = m_data[i] = 0;
      }
      m_innerSize = inners;
    }

    inline LinkedVectorMatrix(int rows, int cols)
      : m_innerSize(0)
    {
      resize(rows, cols);
    }

    template<typename OtherDerived>
    inline LinkedVectorMatrix(const MatrixBase<OtherDerived>& other)
      : m_innerSize(0)
    {
      *this = other.derived();
    }

    inline void swap(LinkedVectorMatrix& other)
    {
      EIGEN_DBG_SPARSE(std::cout << "LinkedVectorMatrix:: swap\n");
      resize(other.rows(), other.cols());
      m_data.swap(other.m_data);
      m_ends.swap(other.m_ends);
    }

    inline LinkedVectorMatrix& operator=(const LinkedVectorMatrix& other)
    {
      if (other.isRValue())
      {
        swap(other.const_cast_derived());
      }
      else
      {
        // TODO implement a specialized deep copy here
        return operator=<LinkedVectorMatrix>(other);
      }
      return *this;
    }

    template<typename OtherDerived>
    inline LinkedVectorMatrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      return SparseMatrixBase<LinkedVectorMatrix>::operator=(other.derived());
    }

  protected:

    // outer vector of inner linked vector chunks
    std::vector<VectorChunk*> m_data;
    // stores a reference to the last vector chunk for efficient filling
    std::vector<VectorChunk*> m_ends;
    int m_innerSize;

};


template<typename Scalar, int _Flags>
class LinkedVectorMatrix<Scalar,_Flags>::InnerIterator
{
  public:

    InnerIterator(const LinkedVectorMatrix& mat, int col)
      : m_matrix(mat), m_el(mat.m_data[col]), m_it(0)
    {}

    InnerIterator& operator++()
    {
      m_it++;
      if (m_it>=m_el->size)
      {
        m_el = m_el->next;
        m_it = 0;
      }
      return *this;
    }

    Scalar value() { return m_el->data[m_it].value; }

    int index() const { return m_el->data[m_it].index; }

    operator bool() const { return m_el && (m_el->next || m_it<m_el->size); }

  protected:
    const LinkedVectorMatrix& m_matrix;
    VectorChunk* m_el;
    int m_it;
};

#endif // EIGEN_LINKEDVECTORMATRIX_H
