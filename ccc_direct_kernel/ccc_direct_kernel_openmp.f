      program ccc_direct_kernel
      use omp_lib
      integer nchi,nchtop,nqmi,nqmf,maxi,maxi2
      integer nchf,ki,i,j,k
      integer nqmfmax
      integer kff,kii
      double precision dotp1,dotp2
      integer gpunum,tnum,ngpus
      integer num_devices
      parameter kmax = 601
      parameter meshr = 15510

      integer, allocatable :: npk(:)      
      double precision, allocatable :: chil(:,:)
     >                  ,chitemp(:,:),vmatt(:,:,:,:)
      double precision, allocatable :: temp3(:,:),temp2(:,:,:),tmp(:,:)

!      integer, external :: omp_get_thread_num,omp_get_max_threads,
!     >                     omp_get_num_devices

      double precision start, finish   

#ifdef GPU
C     set number of gpus on the node
      num_devices=omp_get_num_devices()
      ngpus=max(1,num_devices)
      write(*,*) num_devices
#endif
#ifdef _OPENMP
C     set number of threads
      nnt=omp_get_max_threads()
#endif
    
C     initialisation of job sizes
C     for bigger jobs - change nchtop 
      print '("Initialisation")'
      maxi = meshr
      maxi2 = meshr
      nchi = 1
      nchtop = 415
      nqmf = 117
      nqmi = 117
      nqmfmax = nqmf

C     allocation
      print '("Array allocation")'
      allocate(npk(nchtop+1))
      allocate(chitemp(meshr,nqmi))
      allocate(temp3(meshr,nchi:nchtop))
      allocate(temp2(meshr,nqmi,nchi:nchtop))
      allocate(tmp(nqmi,nqmfmax))
      allocate(vmatt(nqmfmax,nqmi,nchi:nchtop,0:1))
 
      npk(1)=1
      do i = 2, nchtop+1
         npk(i)=npk(i-1)+nqmi
      end do

      allocate(chil(meshr,npk(nchtop+1)-1))

C     generation of random data for chil,temp3,vmatt
      print '("Random data generation")'
      chil(:,:) = 0.01
      temp3(:,:) = 0.5
      temp2(:,:,:) = 0.2
      vmatt(:,:,:,:) = 0.0004
 
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C KERNEL STARTS HERE
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC  

      print '("Kernel started")'

C     for accelerator we can assume that chil and npk have been 
C     copied to the device earlier
#ifdef GPU
      do gpunum=0,ngpus-1
!         call omp_set_default_device(gpunum)
!         call acc_set_device_num(gpunum,acc_device_nvidia)
!$omp target enter data device(gpunum)
!$omp& map(to:chil(1:meshr,1:(npk(nchtop+1)-1)),npk(1:nchtop+1))
      enddo
#endif

!      call cpu_time(start)
      start = omp_get_wtime()

#ifdef GPU
!!$omp parallel num_threads(ngpus)
#else
!!$omp parallel num_threads(nnt)
#endif
!!$omp& private(gpunum,nchf,nqmf,chitemp,ki,kf,i,kff,kii,tmp,tnum)
!!$omp& shared(nchi,nchtop,npk,maxtemp3,temp3,chil,vmatt,ngpus)
!!$omp& shared(nqmi,maxi,maxi2,temp2)

#ifdef GPU
      tnum=omp_get_thread_num()
      gpunum=mod(tnum,ngpus)
      call omp_set_default_device(gpunum)
#endif

#ifdef GPU
!$omp target enter data map(to:vmatt(1:nqmfmax,1:nqmi,nchi:nchtop,0:1),
!$omp& temp3(1:meshr,nchi:nchtop),
!$omp& temp2(1:maxi2,1:nqmi,nchi:nchtop))
!$omp target teams
#endif
      do nchf = nchi, nchtop
         nqmf = npk(nchf+1) - npk(nchf)
#ifdef GPU
!$omp distribute parallel do collapse(2)
#endif
          do ki = 1, nqmi
            do kf = 1, nqmf
               kii = npk(nchi) + ki - 1
               kff = npk(nchf) + kf -1
               dotp1=0
               dotp2=0
#ifdef GPU
!$omp simd reduction(+:dotp1,dotp2)
#endif
               do kd = 1,maxi
                 dotp2=dotp2+chil(kd,kff)*chil(kd,kii)
     >                 *temp3(kd,nchf)
                 dotp1=dotp1+chil(kd,kff)*temp2(kd,ki,nchf)
               end do


               vmatt(kf,ki,nchf,0)=vmatt(kf,ki,nchf,0)+dotp2+dotp1
               vmatt(kf,ki,nchf,1)=vmatt(kf,ki,nchf,0)-2*dotp1
            end do
         end do
       end do
#ifdef GPU
!$omp end target teams
!$omp target exit data map(from:vmatt)
#endif
      finish = omp_get_wtime()
  
      print '("Kernel time: ",f10.5," seconds.")',finish-start

      write(*,*) vmatt(1,1,1,1)

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C KERNEL ENDS HERE
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

#ifdef GPU
      do gpunum=0,ngpus-1
!         call acc_set_device_num(gpunum,acc_device_nvidia)
!         call omp_set_default_device(gpunum)
!$omp target exit data device(gpunum) map(release:chil,npk)
!!map(delete:chil(1:meshr,1:(npk(nchtop+1)-1)),
!!$omp& npk(1:nchtop+1))
      end do
#endif

      deallocate(npk,chitemp,temp3,temp2,tmp,vmatt,chil)

      end program ccc_direct_kernel
