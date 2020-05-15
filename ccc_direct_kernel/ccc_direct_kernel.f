      program ccc_direct_kernel
#ifdef GPU
      use openacc
#endif
      integer nchi,nchtop,nqmi,nqmf,maxi
      integer nchf,ki,i,j,k
      integer nqmfmax,childim
      integer kff,kii
      integer gpunum,tnum,ngpus
      parameter kmax = 601
      parameter meshr = 15510

      integer, allocatable :: npk(:)      
      double precision, allocatable :: chil(:,:,:)
     >                  ,chitemp(:,:),vmatt(:,:,:)
      double precision, allocatable :: temp3(:,:) 

      integer, external :: omp_get_thread_num,omp_get_max_threads

      real :: start, finish   

#ifdef GPU
C     set number of gpus on the node
      ngpus=1
#endif
#ifdef _OPENMP
C     set number of threads
      nnt=omp_get_max_threads()
#endif
    
C     initialisation of job sizes
C     for bigger jobs - change nchtop 
      print '("Initialisation")'
      maxi = meshr
      nchi = 1
      nchtop = 415
      nqmf = 117
      nqmi = 117
      nqmfmax = nqmf
      childim = 1

C     allocation
      print '("Array allocation")'
      allocate(npk(nchtop+1))
      allocate(chitemp(meshr,nqmi))
      allocate(temp3(meshr,nchi:nchtop))
      allocate(vmatt(nqmfmax,nqmi,nchi:nchtop))
 
      npk(1)=1
      do i = 2, nchtop+1
         npk(i)=npk(i-1)+nqmi
      end do

      allocate(chil(meshr,npk(nchtop+1)-1,childim))

C     generation of random data for chil,temp3,vmatt
      print '("Random data generation")'
      chil(:,:,:) = 0.01
      temp3(:,:) = 0.5
      vmatt(:,:,:) = 0.0004
 
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C KERNEL STARTS HERE
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC  

      print '("Kernel started")'

C     for accelerator we can assume that chil and npk have been 
C     copied to the device earlier
#ifdef GPU
      do gpunum=0,ngpus-1
         call acc_set_device_num(gpunum,acc_device_nvidia)
!$acc enter data copyin(chil(1:meshr,1:(npk(nchtop+1)-1),1))
!$acc& copyin(npk(1:nchtop+1))
      enddo
#endif

      call cpu_time(start)

C#ifdef GPU
!$omp parallel num_threads(ngpus)
C#else
C!$omp parallel num_threads(nnt)
C#endif
!$omp& private(gpunum,nchf,nqmf,chitemp,ki,kf,i,kff,kii)
!$omp& shared(nchi,nchtop,npk,maxtemp3,temp3,chil,vmatt,ngpus)
!$omp& shared(nqmi,maxi)

C#ifdef GPU
      tnum=omp_get_thread_num()
      gpunum=mod(tnum,ngpus)
      call acc_set_device_num(gpunum,acc_device_nvidia)
C#endif

!$acc data
!$acc& copyin(vmatt(1:nqmfmax,1:nqmi,nchi:nchtop))
!$acc& present(npk(1:nchtop+1))
!$acc& present(chil(1:meshr,1:(npk(nchtop+1)-1),1:childim))
!!$acc& present(nchtop)
!$acc& copyin(nqmi,temp3(1:meshr,nchi:nchtop))
!$acc& create(chitemp)
!$omp do schedule(dynamic)
      do nchf = nchi, nchtop
         nqmf = npk(nchf+1) - npk(nchf)
!$acc kernels 
!$acc loop independent collapse(2)
         do ki = 1, nqmi
            do i = 1, maxi !minki, maxi
               chitemp(i,ki) = temp3(i,nchf) * chil(i,ki+npk(nchi)-1,1)
            enddo
         enddo
!$acc loop independent collapse(2)
         do ki = 1, nqmi
            do kf=1,nqmf
               kii = npk(nchi) + ki - 1
               kff = npk(nchf) + kf - 1
               if (kff.ge.kii) then
               vmatt(kf,ki,nchf)=vmatt(kf,ki,nchf)+dot_product(
     >           chil(1:maxi,kf+npk(nchf)-1,1),chitemp(1:maxi,ki))
               endif
            end do
         end do
!$acc end kernels
!$acc update self(vmatt(1:nqmf,1:nqmi,nchf)) async
       end do
!$omp end do

!$acc wait

!$acc end data
!$omp end parallel

      call cpu_time(finish)
  
      print '("Kernel time: ",f6.3," seconds.")',finish-start

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C KERNEL ENDS HERE
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

#ifdef GPU
      do gpunum=0,ngpus-1
         call acc_set_device_num(gpunum,acc_device_nvidia)
!$acc exit data delete(chil,npk)
      end do
#endif

      deallocate(npk,chitemp,temp3,vmatt,chil)

      end program ccc_direct_kernel
