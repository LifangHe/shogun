/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Parijat Mazumdar
 */

#include "shogun/clustering/KMeansLloydImpl.h"
#include <shogun/distance/Distance.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/linalg.h>

using namespace shogun;

namespace shogun
{
void CKMeansLloydImpl::Lloyd_KMeans(int32_t k, CDistance* distance, int32_t max_iter, SGMatrix<float64_t> mus,
		SGVector<int32_t> ClList, SGVector<float64_t> weights_set, bool fixed_centers)
{
	CDenseFeatures<float64_t>* lhs=
		CDenseFeatures<float64_t>::obtain_from_generic(distance->get_lhs());
	
	int32_t XSize=lhs->get_num_vectors();
	int32_t dimensions=lhs->get_num_features();

	CDenseFeatures<float64_t>* rhs_mus=new CDenseFeatures<float64_t>(0);
	CFeatures* rhs_cache=distance->replace_rhs(rhs_mus);

	SGVector<float64_t> dists=SGVector<float64_t>(k*XSize);
	dists.zero();
	
//	SGMatrix<float64_t>lh_m=lhs->get_feature_matrix();
	SGVector<float64_t> lhs_sq_norm = SGVector<float64_t>(XSize);

//	SGMatrix<float64_t> sq = linalg::elementwise_square(lh_m);
//	linalg::colwise_sum(sq, rhs_sq_norm);
	for(int32_t idx_i =0; idx_i<XSize; idx_i++)
	{
		SGVector<float64_t> avec=((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_i);
		float64_t temp = linalg::dot(avec, avec);
		lhs_sq_norm[idx_i] = temp;
		((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_i);
		
	}
	((CEuclideanDistance*) distance)->set_lhs_sq_norm(lhs_sq_norm);

	int32_t changed=1;
	int32_t iter=0;
	int32_t vlen=0;
	bool vfree=false;
	float64_t* vec=NULL;

	while (changed && (iter<max_iter))
	{
		iter++;
		if (iter==max_iter-1)
			SG_SWARNING("kmeans clustering changed throughout %d iterations stopping...\n", max_iter-1)

		changed=0;

		rhs_mus->copy_feature_matrix(mus);

		//SGMatrix<float64_t> dist = ((CEuclideanDistance*) distance)->get_precomputed_distance();

	
		SGVector<float64_t> rhs_sq_norm(k);
		for(int32_t idx_i =0; idx_i<k; idx_i++)
		{
			SGVector<float64_t> tempvec=((CDenseFeatures<float64_t>*) rhs_mus)->get_feature_vector(idx_i);
			float64_t temp = linalg::dot(tempvec, tempvec);
			rhs_sq_norm[idx_i] = temp;
			((CDenseFeatures<float64_t>*) rhs_mus)->free_feature_vector(tempvec, idx_i);
			
		}
		((CEuclideanDistance*) distance)->set_rhs_sq_norm(rhs_sq_norm);
		
		for (int32_t i=0; i<XSize; i++)
		{ 
			
			const int32_t Pat=i;
			const int32_t ClList_Pat=ClList[Pat];
			int32_t imini, j;
			float64_t mini;
	

			for(int32_t idx_k=0;idx_k<k;idx_k++)
			{
				dists[idx_k]=distance->distance(Pat,idx_k);
			}
//			for(int32_t idx_k=0;idx_k<k;idx_k++)
//				dists[idx_k]=dist(idx_k, Pat);

//			SGVector<float64_t> dists(k);
//			dists = ((CEuclideanDistance*) distance)->multiple_distance(i);
			/* [mini,imini]=min(dists(:,i)) ; */
			imini=0 ; mini=dists[0];
			for (j=1; j<k; j++)
				if (dists[j]<mini)
				{
					mini=dists[j];
					imini=j;
				}
			if (imini!=ClList_Pat)
			{
				changed++;

				/* weights_set(imini) = weights_set(imini) + 1.0 ; */
				weights_set[imini]+= 1.0;
				/* weights_set(j)     = weights_set(j)     - 1.0 ; */
				if (iter != 1)
					weights_set[ClList_Pat]-= 1.0;

				ClList[Pat] = imini;
			}
			else if (iter == 1)
			{
				weights_set[ClList_Pat]+=1.0;
			}
					
		}
		if (!fixed_centers)
		{
			
			/* mus=zeros(dimensions, k) ; */
			mus.zero();
			for (int32_t i=0; i<XSize; i++)
			{
				int32_t Cl=ClList[i];

				vec=lhs->get_feature_vector(i, vlen, vfree);

				for (int32_t j=0; j<dimensions; j++)
					mus.matrix[Cl*dimensions+j] += vec[j];
				lhs->free_feature_vector(vec, i, vfree);
			}
//			SGMatrix<float64_t> weights(k, dimensions);//=SGMatrix<float64_t>(dimensions, k);
		//	weights.transpose_matrix();
//			linalg::set_rows_const(weights, weights_set);
//			SGMatrix<float64_t>::transpose_matrix(weights.matrix, weights.num_rows, weights.num_cols);
//			linalg::elementwise_compute_inplace(weights, [](float64_t& w)
//			{
//				return 1/w;  
//			});
//
//			linalg::elementwise_product(mus, weights, mus);
		
			for (int32_t i=0; i<k; i++)
			{
				if (weights_set[i]!=0.0)
				{
					for (int32_t j=0; j<dimensions; j++)
						mus.matrix[i*dimensions+j] /= weights_set[i];
				}
			}
					
		}
		((CEuclideanDistance*) distance)->reset_rhs_sq_norm();
		
		
		if (iter%1000 == 0)
			SG_SINFO("Iteration[%d/%d]: Assignment of %i patterns changed.\n", iter, max_iter, changed)
			SG_SPRINT("iter %d, %i", iter, changed)
	}
	((CEuclideanDistance*) distance)->reset_lhs_sq_norm();
	distance->replace_rhs(rhs_cache);
	delete rhs_mus;
	SG_UNREF(lhs);
}
}
