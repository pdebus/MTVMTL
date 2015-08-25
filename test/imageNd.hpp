#ifndef VPP_IMAGENd_HPP__
# define VPP_IMAGENd_HPP__

# include <iod/sio.hh>
# include <vpp/core/imageNd.hh>
# include <vpp/core/boxNd.hh>
# include <vpp/core/symbols.hh>

# ifndef VPP_DEFAULT_IMAGE_ALIGNMENT

# ifdef __AVX2__
#  define VPP_DEFAULT_IMAGE_ALIGNMENT 32
# else
#  define VPP_DEFAULT_IMAGE_ALIGNMENT 16
# endif

# endif

namespace vpp
{

  template <typename V, unsigned N>
  imageNd<V, N>::imageNd()
  {
  }

  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(const std::initializer_list<int>& dims, const O&... options)
  {
    allocate(dims, iod::D(options...));
  }

  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(const std::vector<int>& dims, const O&... options)
  {
    allocate(dims, iod::D(options...));
  }


  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(int ncols, const O&... options)
    : imageNd(make_box1d(ncols), options...)
  {
    static_assert(N == 1, "ImageNd constructor: bad dimension.");
  }

  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(int nrows, int ncols, const O&... options)
    : imageNd(make_box2d(nrows, ncols), options...)
  {
    static_assert(N == 2, "ImageNd constructor: bad dimension.");
  }

  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(int nslices, int nrows, int ncols, const O&... options)
    : imageNd(make_box3d(nslices, nrows, ncols), options...)
  {
    static_assert(N == 3, "ImageNd constructor: bad dimension.");
  }

  template <typename V, unsigned N>
  imageNd<V, N>::imageNd(imageNd<V, N>&& other)
    : ptr_(std::move(other.ptr_))
  {
  }


  template <typename V, unsigned N>
  imageNd<V, N>& imageNd<V, N>::operator=(const imageNd<V, N>& other)
  {
    ptr_ = other.ptr_;
    return *this;
  }

  template <typename V, unsigned N>
  imageNd<V, N>::imageNd(const imageNd<V, N>& other)
  {
    *this = other;
  }


  template <typename V, unsigned N>
  imageNd<V, N>& imageNd<V, N>::operator=(imageNd<V, N>&& other)
  {
    ptr_ = std::move(other.ptr_);
    return *this;
  }

  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(const boxNd<N>& domain, const O&... _options)
  {
    std::vector<int> dims(N);

    for (int i = 0; i < int(N); i++)
	dims[i] = domain.size(i);


    const auto options = iod::D(_options...);

    static_assert(
      !options.has(_data) or options.has(_pitch),
      "You must provide the pitch (Number of bits between the begining of to successive lines when providing a data pointer the image constructor.");

    if (options.has(_data))
    {
      ptr_ = std::shared_ptr<imageNd_data<V, N>>(new imageNd_data<V, N>());

      ptr_->data_ = (V*) options.get(_data, (V*)0);
      ptr_->begin_ = ptr_->data_;
      ptr_->pitch_ = options.get(_pitch, 0);
      ptr_->domain_ = domain;
      ptr_->border_ = options.get(_border, 0);

      int size = ptr_->pitch_;
      for (int n = 0; n < N - 1; n++)
        size *= ptr_->domain_.size(n);

      ptr_->data_end_ = (V*)((char*) ptr_->data_ + size);
    }
    else
      allocate(dims, options);
  }

  template <typename V, unsigned N>
  imageNd<V, N>::~imageNd()
  {
    // The shared_ptr this_->ptr_ will destroy the data if needed.
  }

  template <typename V, unsigned N>
  template <typename... O>
  void imageNd<V, N>::allocate(const std::vector<int>& dims, const iod::sio<O...>& options)
  {
    const int align_size = options.get(_aligned, VPP_DEFAULT_IMAGE_ALIGNMENT); // Memory alignment of rows.

    typedef unsigned long long ULL;
    ptr_ = std::make_shared<imageNd_data<V, N>>();
    auto& d = *ptr_;
    d.border_ = options.get(_border, 0);

    int border_size = d.border_ * sizeof(V);
    int border_padding = 0;
    if (border_size % align_size)
    {
      border_padding = align_size - (border_size % align_size);
      border_size += border_padding;
    }

    d.strides_ = std::vector<int>(N);
    for (int i=0; i< N; i++)
	d.strides_[i] = dims[i];

    d.pitch_ = dims[N - 1] * sizeof(V) + border_size * 2;
    if (d.pitch_ % align_size) d.pitch_ += align_size - (d.pitch_ % align_size);

    int size = 1;
    for (int i = 0; i < N - 1; i++)
      size *= (dims[i] + border_size * 2);
    size *= d.pitch_;
    
    d.data_ = (V*) malloc(align_size + size);
    d.data_sptr_ = std::shared_ptr<void>(d.data_, [] (V* p) { 
        free((char*)p); 
      });

    if (ULL(d.data_) % align_size)
      d.data_ = (V*)((char*)(d.data_) + align_size - int(ULL(d.data_) % align_size));

    d.data_end_ = d.data_ + size / sizeof(V);
    assert(!(ULL(d.data_) % align_size));
    d.domain_.p1() = vint<N>::Zero();
    for (unsigned i = 0; i < N; i++)
      d.domain_.p2()[i] = dims[i] - 1;

    // Set begin_, the address of the first pixel.
    vint<N> p = vint<N>::Ones() * d.border_;
    d.begin_ = (V*)((char*)d.data_ + border_padding + coords_to_offset(p));

  }

  template <typename V, unsigned N>
  int imageNd<V, N>::coords_to_offset(const vint<N>& p) const
  {
    int row_idx = p[N-2];
    int ds = 1;
    for (int i = N - 3; i >= 0; i--)
    {
      ds *= ptr_->strides_[i + 1];
      row_idx += ds * p[i];
    }
    return row_idx * ptr_->pitch_ + p[N - 1] * sizeof(V);
  }

  template <typename V, unsigned N>
  V&
  imageNd<V, N>::operator()(const vint<N>& p)
  {
    assert(domain_with_border().has(p));
    V* addr = (V*)((char*)ptr_->begin_ + coords_to_offset(p));
    assert(addr < ptr_->data_end_);
    return *addr;
  }

  template <typename V, unsigned N>
  const V&
  imageNd<V, N>::operator()(const vint<N>& p) const
  {
    assert(domain_with_border().has(p));
    const V* addr = (V*)((char*)ptr_->begin_ + coords_to_offset(p));
    assert(addr < ptr_->data_end_);
    return *addr;
  }

  template <typename V, unsigned N>
  V
  imageNd<V, N>::linear_interpolate(const vfloat<N>& p) const
  {
    static_assert(N == 2, "linear_interpolate only supports 2d images.");

    vint2 x = p.template cast<int>();
    float a0 = p[0] - x[0];
    float a1 = p[1] - x[1];

    const V* l1 = address_of(x);
    const V* l2 = (const V*)((const char*)l1 + ptr_->pitch_);

    //typedef plus_promotion<V> S;
    typedef cast_to_float<V> S;

    return vpp::cast<V>((1 - a0) * (1 - a1) *  vpp::cast<S>(l1[0]) +
                   a0 * (1 - a1) *  vpp::cast<S>(l2[0]) +
                   (1 - a0) * a1 *  vpp::cast<S>(l1[1]) +
                   a0 * a1 *  vpp::cast<S>(l2[1]));
  }

  template <typename V, unsigned N>
  V*
  imageNd<V, N>::address_of(const vint<N>& p)
  {
    return (V*)((char*)(ptr_->begin_) + coords_to_offset(p));
  }

  template <typename V, unsigned N>
  const V*
  imageNd<V, N>::address_of(const vint<N>& p) const
  {
    return (V*)((char*)(ptr_->begin_) + coords_to_offset(p));
  }

  template <typename V, unsigned N>
  int
  imageNd<V, N>::offset_of(const vint<N>& p) const
  {
    return coords_to_offset(p);
  }

  template <typename V, unsigned N>
  imageNd<V, N>
  imageNd<V, N>::subimage(const boxNd<N>& d)
  {
    imageNd<V, N> res;
    res.ptr_ = std::shared_ptr<imageNd_data<V, N>>(new imageNd_data<V, N>());
    *res.ptr_.get() = *this->ptr_.get(); // Copy the whole image data.
    res.ptr_->begin_ = address_of(d.p1());
    boxNd<N> domain = d;
    domain.p2() -= domain.p1();
    domain.p1() -= domain.p1();
    res.ptr_->domain_ = domain;
    return res;
  }

  template <typename V, unsigned N>
  const imageNd<V, N>
  imageNd<V, N>::const_subimage(const boxNd<N>& d) const
  {
    imageNd<V, N> res;

    res.ptr_ = std::shared_ptr<imageNd_data<V, N>>(new imageNd_data<V, N>());
    *res.ptr_.get() = *this->ptr_.get();
    res.ptr_->begin_ = const_cast<V*>(address_of(d.p1()));
    boxNd<N> domain = d;
    domain.p2() -= domain.p1();
    domain.p1() -= domain.p1();
    res.ptr_->domain_ = domain;
    return res;
  }

};

#endif
