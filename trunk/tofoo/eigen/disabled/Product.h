// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_PRODUCT_H
#define EIGEN_PRODUCT_H

template<int Index, int Size, typename Lhs, typename Rhs>
struct ei_product_unroller
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                               typename Lhs::Scalar &res)
  {
    ei_product_unroller<Index-1, Size, Lhs, Rhs>::run(row, col, lhs, rhs, res);
    res += lhs.coeff(row, Index) * rhs.coeff(Index, col);
  }
};

template<int Size, typename Lhs, typename Rhs>
struct ei_product_unroller<0, Size, Lhs, Rhs>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                               typename Lhs::Scalar &res)
  {
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
  }
};

template<int Index, typename Lhs, typename Rhs>
struct ei_product_unroller<Index, Dynamic, Lhs, Rhs>
{
  inline static void run(int, int, const Lhs&, const Rhs&, typename Lhs::Scalar&) {}
};

// prevent buggy user code from causing an infinite recursion
template<int Index, typename Lhs, typename Rhs>
struct ei_product_unroller<Index, 0, Lhs, Rhs>
{
  inline static void run(int, int, const Lhs&, const Rhs&, typename Lhs::Scalar&) {}
};

template<typename Lhs, typename Rhs>
struct ei_product_unroller<0, Dynamic, Lhs, Rhs>
{
  static void run(int, int, const Lhs&, const Rhs&, typename Lhs::Scalar&) {}
};

template<bool RowMajor, int Index, int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller;

template<int Index, int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<true, Index, Size, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_packet_product_unroller<true, Index-1, Size, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(ei_pset1(lhs.coeff(row, Index)), rhs.template packet<Aligned>(Index, col), res);
  }
};

template<int Index, int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<false, Index, Size, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_packet_product_unroller<false, Index-1, Size, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(lhs.template packet<Aligned>(row, Index), ei_pset1(rhs.coeff(Index, col)), res);
  }
};

template<int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<true, 0, Size, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(ei_pset1(lhs.coeff(row, 0)),rhs.template packet<Aligned>(0, col));
  }
};

template<int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<false, 0, Size, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(lhs.template packet<Aligned>(row, 0), ei_pset1(rhs.coeff(0, col)));
  }
};

template<bool RowMajor, int Index, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<RowMajor, Index, Dynamic, Lhs, Rhs, PacketScalar>
{
  inline static void run(int, int, const Lhs&, const Rhs&, PacketScalar&) {}
};

template<int Index, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<false, Index, Dynamic, Lhs, Rhs, PacketScalar>
{
  inline static void run(int, int, const Lhs&, const Rhs&, PacketScalar&) {}
};

template<typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<false, 0, Dynamic, Lhs, Rhs, PacketScalar>
{
  static void run(int, int, const Lhs&, const Rhs&, PacketScalar&) {}
};

template<typename Product, bool RowMajor = true> struct ProductPacketImpl {
  inline static typename Product::PacketScalar execute(const Product& product, int row, int col)
  { return product._packetRowMajor(row,col); }
};

template<typename Product> struct ProductPacketImpl<Product, false> {
  inline static typename Product::PacketScalar execute(const Product& product, int row, int col)
  { return product._packetColumnMajor(row,col); }
};

/** \class Product
  *
  * \brief Expression of the product of two matrices
  *
  * \param Lhs the type of the left-hand side
  * \param Rhs the type of the right-hand side
  * \param EvalMode internal use only
  *
  * This class represents an expression of the product of two matrices.
  * It is the return type of the operator* between matrices, and most of the time
  * this is the only way it is used.
  *
  * \sa class Sum, class Difference
  */
template<typename Lhs, typename Rhs> struct ei_product_eval_mode
{
  enum{ value =  Lhs::MaxRowsAtCompileTime >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
              && Rhs::MaxColsAtCompileTime >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
              && (!( (Lhs::Flags&RowMajorBit) && ((Rhs::Flags&RowMajorBit) ^ RowMajorBit)))
              ? CacheFriendlyProduct : NormalProduct };
};

template<typename Lhs, typename Rhs, int EvalMode>
struct ei_traits<Product<Lhs, Rhs, EvalMode> >
{
  typedef typename Lhs::Scalar Scalar;
  typedef typename ei_nested<Lhs,Rhs::ColsAtCompileTime>::type LhsNested;
  typedef typename ei_nested<Rhs,Lhs::RowsAtCompileTime>::type RhsNested;
  typedef typename ei_unref<LhsNested>::type _LhsNested;
  typedef typename ei_unref<RhsNested>::type _RhsNested;
  enum {
    LhsCoeffReadCost = _LhsNested::CoeffReadCost,
    RhsCoeffReadCost = _RhsNested::CoeffReadCost,
    LhsFlags = _LhsNested::Flags,
    RhsFlags = _RhsNested::Flags,
    RowsAtCompileTime = Lhs::RowsAtCompileTime,
    ColsAtCompileTime = Rhs::ColsAtCompileTime,
    MaxRowsAtCompileTime = Lhs::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = Rhs::MaxColsAtCompileTime,
    _RhsPacketAccess = (RhsFlags & RowMajorBit) && (RhsFlags & PacketAccessBit) && (ColsAtCompileTime % ei_packet_traits<Scalar>::size == 0),
    _LhsPacketAccess = (!(LhsFlags & RowMajorBit)) && (LhsFlags & PacketAccessBit) && (RowsAtCompileTime % ei_packet_traits<Scalar>::size == 0),
    _PacketAccess = (_LhsPacketAccess || _RhsPacketAccess) ? 1 : 0,
    _RowMajor = (RhsFlags & RowMajorBit)
              && (EvalMode==(int)CacheFriendlyProduct ? (int)LhsFlags & RowMajorBit : (!_LhsPacketAccess)),
    _LostBits = HereditaryBits & ~(
                (_RowMajor ? 0 : RowMajorBit)
              | ((RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic) ? 0 : LargeBit)),
    Flags = ((unsigned int)(LhsFlags | RhsFlags) & _LostBits)
          | EvalBeforeAssigningBit
          | EvalBeforeNestingBit
          | (_PacketAccess ? PacketAccessBit : 0),
    CoeffReadCost
      = Lhs::ColsAtCompileTime == Dynamic
      ? Dynamic
      : Lhs::ColsAtCompileTime
        * (NumTraits<Scalar>::MulCost + LhsCoeffReadCost + RhsCoeffReadCost)
        + (Lhs::ColsAtCompileTime - 1) * NumTraits<Scalar>::AddCost
  };
};

template<typename Lhs, typename Rhs, int EvalMode> class Product : ei_no_assignment_operator,
  public MatrixBase<Product<Lhs, Rhs, EvalMode> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Product)
    friend class ProductPacketImpl<Product,Flags&RowMajorBit>;
    typedef typename ei_traits<Product>::LhsNested LhsNested;
    typedef typename ei_traits<Product>::RhsNested RhsNested;
    typedef typename ei_traits<Product>::_LhsNested _LhsNested;
    typedef typename ei_traits<Product>::_RhsNested _RhsNested;

    inline Product(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      ei_assert(lhs.cols() == rhs.rows());
    }

    /** \internal */
    template<typename DestDerived, int AlignedMode>
    void _cacheOptimalEval(DestDerived& res, ei_meta_false) const;
    #ifdef EIGEN_VECTORIZE
    template<typename DestDerived, int AlignedMode>
    void _cacheOptimalEval(DestDerived& res, ei_meta_true) const;
    #endif

  private:

    inline int _rows() const { return m_lhs.rows(); }
    inline int _cols() const { return m_rhs.cols(); }

    const Scalar _coeff(int row, int col) const
    {
      Scalar res;
      const bool unroll = CoeffReadCost <= EIGEN_UNROLLING_LIMIT;
      if(unroll)
      {
        ei_product_unroller<Lhs::ColsAtCompileTime-1,
                            unroll ? Lhs::ColsAtCompileTime : Dynamic,
                            _LhsNested, _RhsNested>
          ::run(row, col, m_lhs, m_rhs, res);
      }
      else
      {
        res = m_lhs.coeff(row, 0) * m_rhs.coeff(0, col);
        for(int i = 1; i < m_lhs.cols(); i++)
          res += m_lhs.coeff(row, i) * m_rhs.coeff(i, col);
      }
      return res;
    }

    template<int LoadMode>
    const PacketScalar _packet(int row, int col) const
    {
      if(Lhs::ColsAtCompileTime <= EIGEN_UNROLLING_LIMIT)
      {
        PacketScalar res;
        ei_packet_product_unroller<Flags&RowMajorBit, Lhs::ColsAtCompileTime-1,
                            Lhs::ColsAtCompileTime <= EIGEN_UNROLLING_LIMIT
                              ? Lhs::ColsAtCompileTime : Dynamic,
                            _LhsNested, _RhsNested, PacketScalar>
          ::run(row, col, m_lhs, m_rhs, res);
        return res;
      }
      else
        return ProductPacketImpl<Product,Flags&RowMajorBit>::execute(*this, row, col);
    }

    const PacketScalar _packetRowMajor(int row, int col) const
    {
      PacketScalar res;
      res = ei_pmul(ei_pset1(m_lhs.coeff(row, 0)),m_rhs.template packet<Aligned>(0, col));
      for(int i = 1; i < m_lhs.cols(); i++)
        res =  ei_pmadd(ei_pset1(m_lhs.coeff(row, i)), m_rhs.template packet<Aligned>(i, col), res);
      return res;
    }

    const PacketScalar _packetColumnMajor(int row, int col) const
    {
      PacketScalar res;
      res = ei_pmul(m_lhs.template packet<Aligned>(row, 0), ei_pset1(m_rhs.coeff(0, col)));
      for(int i = 1; i < m_lhs.cols(); i++)
        res =  ei_pmadd(m_lhs.template packet<Aligned>(row, i), ei_pset1(m_rhs.coeff(i, col)), res);
      return res;
//       const PacketScalar tmp[4];
//       ei_punpack(m_rhs.packet(0,col), tmp);
//
//       return
//         ei_pmadd(m_lhs.packet(row, 0), tmp[0],
//           ei_pmadd(m_lhs.packet(row, 1), tmp[1],
//             ei_pmadd(m_lhs.packet(row, 2), tmp[2]
//               ei_pmul(m_lhs.packet(row, 3), tmp[3]))));
    }


  protected:
    const LhsNested m_lhs;
    const RhsNested m_rhs;
};

/** \returns the matrix product of \c *this and \a other.
  *
  * \note This function causes an immediate evaluation. If you want to perform a matrix product
  * without immediate evaluation, call .lazy() on one of the matrices before taking the product.
  *
  * \sa lazy(), operator*=(const MatrixBase&)
  */
template<typename Derived>
template<typename OtherDerived>
inline const Product<Derived,OtherDerived>
MatrixBase<Derived>::operator*(const MatrixBase<OtherDerived> &other) const
{
  return Product<Derived,OtherDerived>(derived(), other.derived());
}

/** replaces \c *this by \c *this * \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Derived>
template<typename OtherDerived>
inline Derived &
MatrixBase<Derived>::operator*=(const MatrixBase<OtherDerived> &other)
{
  return *this = *this * other;
}

template<typename Derived>
template<typename Lhs, typename Rhs>
inline Derived& MatrixBase<Derived>::lazyAssign(const Product<Lhs,Rhs,CacheFriendlyProduct>& product)
{
  product.template _cacheOptimalEval<Derived, Aligned>(derived(),
    #ifdef EIGEN_VECTORIZE
    typename ei_meta_if<Flags & PacketAccessBit, ei_meta_true, ei_meta_false>::ret()
    #else
    ei_meta_false()
    #endif
    );
  return derived();
}

template<typename Lhs, typename Rhs, int EvalMode>
template<typename DestDerived, int AlignedMode>
void Product<Lhs,Rhs,EvalMode>::_cacheOptimalEval(DestDerived& res, ei_meta_false) const
{
  res.setZero();
  const int cols4 = m_lhs.cols() & 0xfffffffC;
  if (Lhs::Flags&RowMajorBit)
  {
//     std::cout << "opt rhs\n";
    int j=0;
    for(; j<cols4; j+=4)
    {
      for(int k=0; k<this->rows(); ++k)
      {
        const Scalar tmp0 = m_lhs.coeff(k,j  );
        const Scalar tmp1 = m_lhs.coeff(k,j+1);
        const Scalar tmp2 = m_lhs.coeff(k,j+2);
        const Scalar tmp3 = m_lhs.coeff(k,j+3);
        for (int i=0; i<this->cols(); ++i)
          res.coeffRef(k,i) += tmp0 * m_rhs.coeff(j+0,i) + tmp1 * m_rhs.coeff(j+1,i)
                             + tmp2 * m_rhs.coeff(j+2,i) + tmp3 * m_rhs.coeff(j+3,i);
      }
    }
    for(; j<m_lhs.cols(); ++j)
    {
      for(int k=0; k<this->rows(); ++k)
      {
        const Scalar tmp = m_rhs.coeff(k,j);
        for (int i=0; i<this->cols(); ++i)
          res.coeffRef(k,i) += tmp * m_lhs.coeff(j,i);
      }
    }
  }
  else
  {
//     std::cout << "opt lhs\n";
    int j = 0;
    for(; j<cols4; j+=4)
    {
      for(int k=0; k<this->cols(); ++k)
      {
        const Scalar tmp0 = m_rhs.coeff(j  ,k);
        const Scalar tmp1 = m_rhs.coeff(j+1,k);
        const Scalar tmp2 = m_rhs.coeff(j+2,k);
        const Scalar tmp3 = m_rhs.coeff(j+3,k);
        for (int i=0; i<this->rows(); ++i)
          res.coeffRef(i,k) += tmp0 * m_lhs.coeff(i,j+0) + tmp1 * m_lhs.coeff(i,j+1)
                             + tmp2 * m_lhs.coeff(i,j+2) + tmp3 * m_lhs.coeff(i,j+3);
      }
    }
    for(; j<m_lhs.cols(); ++j)
    {
      for(int k=0; k<this->cols(); ++k)
      {
        const Scalar tmp = m_rhs.coeff(j,k);
        for (int i=0; i<this->rows(); ++i)
          res.coeffRef(i,k) += tmp * m_lhs.coeff(i,j);
      }
    }
  }
}

#ifdef EIGEN_VECTORIZE
template<typename Lhs, typename Rhs, int EvalMode>
template<typename DestDerived, int AlignedMode>
void Product<Lhs,Rhs,EvalMode>::_cacheOptimalEval(DestDerived& res, ei_meta_true) const
{

  if (((Lhs::Flags&RowMajorBit) && (_cols() % ei_packet_traits<Scalar>::size != 0))
    || (_rows() % ei_packet_traits<Scalar>::size != 0))
  {
    return _cacheOptimalEval<DestDerived, AlignedMode>(res, ei_meta_false());
  }

  res.setZero();
  const int cols4 = m_lhs.cols() & 0xfffffffC;
  if (Lhs::Flags&RowMajorBit)
  {
//     std::cout << "packet rhs\n";
    int j=0;
    for(; j<cols4; j+=4)
    {
      for(int k=0; k<this->rows(); k++)
      {
        const typename ei_packet_traits<Scalar>::type tmp0 = ei_pset1(m_lhs.coeff(k,j+0));
        const typename ei_packet_traits<Scalar>::type tmp1 = ei_pset1(m_lhs.coeff(k,j+1));
        const typename ei_packet_traits<Scalar>::type tmp2 = ei_pset1(m_lhs.coeff(k,j+2));
        const typename ei_packet_traits<Scalar>::type tmp3 = ei_pset1(m_lhs.coeff(k,j+3));
        for (int i=0; i<this->cols(); i+=ei_packet_traits<Scalar>::size)
        {
          res.template writePacket<AlignedMode>(k,i,
            ei_pmadd(tmp0, m_rhs.template packet<AlignedMode>(j+0,i),
              ei_pmadd(tmp1, m_rhs.template packet<AlignedMode>(j+1,i),
                ei_pmadd(tmp2, m_rhs.template packet<AlignedMode>(j+2,i),
                  ei_pmadd(tmp3, m_rhs.template packet<AlignedMode>(j+3,i),
                    res.template packet<AlignedMode>(k,i)))))
          );
        }
      }
    }
    for(; j<m_lhs.cols(); ++j)
    {
      for(int k=0; k<this->rows(); k++)
      {
        const typename ei_packet_traits<Scalar>::type tmp = ei_pset1(m_lhs.coeff(k,j));
        for (int i=0; i<this->cols(); i+=ei_packet_traits<Scalar>::size)
          res.template writePacket<AlignedMode>(k,i,
            ei_pmadd(tmp, m_rhs.template packet<AlignedMode>(j,i), res.template packet<AlignedMode>(k,i)));
      }
    }
  }
  else
  {
//     std::cout << "packet lhs\n";
    int k=0;
    for(; k<cols4; k+=4)
    {
      for(int j=0; j<this->cols(); j+=1)
      {
        const typename ei_packet_traits<Scalar>::type tmp0 = ei_pset1(m_rhs.coeff(k+0,j));
        const typename ei_packet_traits<Scalar>::type tmp1 = ei_pset1(m_rhs.coeff(k+1,j));
        const typename ei_packet_traits<Scalar>::type tmp2 = ei_pset1(m_rhs.coeff(k+2,j));
        const typename ei_packet_traits<Scalar>::type tmp3 = ei_pset1(m_rhs.coeff(k+3,j));

        for (int i=0; i<this->rows(); i+=ei_packet_traits<Scalar>::size)
        {
          res.template writePacket<AlignedMode>(i,j,
            ei_pmadd(tmp0, m_lhs.template packet<AlignedMode>(i,k),
              ei_pmadd(tmp1, m_lhs.template packet<AlignedMode>(i,k+1),
                ei_pmadd(tmp2, m_lhs.template packet<AlignedMode>(i,k+2),
                  ei_pmadd(tmp3, m_lhs.template packet<AlignedMode>(i,k+3),
                    res.template packet<AlignedMode>(i,j)))))
          );
        }
      }
    }
    for(; k<m_lhs.cols(); ++k)
    {
      for(int j=0; j<this->cols(); j++)
      {
        const typename ei_packet_traits<Scalar>::type tmp = ei_pset1(m_rhs.coeff(k,j));
        for (int i=0; i<this->rows(); i+=ei_packet_traits<Scalar>::size)
          res.template writePacket<AlignedMode>(k,j,
            ei_pmadd(tmp, m_lhs.template packet<AlignedMode>(i,k), res.template packet<AlignedMode>(i,j)));
      }
    }
  }
}
#endif // EIGEN_VECTORIZE

#endif // EIGEN_PRODUCT_H
