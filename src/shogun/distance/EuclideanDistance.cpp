/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/mathematics/linalg/linalg.h>

using namespace shogun;

CEuclideanDistance::CEuclideanDistance() : CRealDistance()
{
	init();
}

CEuclideanDistance::CEuclideanDistance(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r)
: CRealDistance()
{
	init();
	init(l, r);
}

CEuclideanDistance::~CEuclideanDistance()
{
	cleanup();
}

bool CEuclideanDistance::init(CFeatures* l, CFeatures* r)
{
	CRealDistance::init(l, r);

	return true;
}

void CEuclideanDistance::cleanup()
{
}

float64_t CEuclideanDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;
	float64_t result=0;

	float64_t* avec=((CDenseFeatures<float64_t>*) lhs)->
		get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=((CDenseFeatures<float64_t>*) rhs)->
		get_feature_vector(idx_b, blen, bfree);

//	SGVector<float64_t> avec=((CDenseFeatures<float64_t>*) lhs)->
//		get_feature_vector(idx_a);
//	SGVector<float64_t> bvec=((CDenseFeatures<float64_t>*) rhs)->
//		get_feature_vector(idx_b);
	ASSERT(alen==blen);
	if(m_dot_enabled)
	{
		result+=CMath::dot(avec, bvec, alen);
		result*=-2;
		if(m_rhs_sq_norms)
		{
			result+=m_rhs_squared_norms[idx_b];
		}
		else
		{
			result+=CMath::dot(bvec, bvec, alen);
		}
		if(m_lhs_sq_norms)
		{
			result+=m_lhs_squared_norms[idx_a];
		}
		else
		{
			result+=CMath::dot(avec, avec, alen);
		}
		if (disable_sqrt)
			return result;
	
		return CMath::sqrt(result);
	}

	for (int32_t i=0; i<alen; i++)
		result+=CMath::sq(avec[i] - bvec[i]);

//	result += linalg::dot<linalg::Backend::EIGEN3>(avec, bvec);
//	result *= -2.0;
//	result += linalg::dot<linalg::Backend::EIGEN3>(avec, avec);
//	result += linalg::dot<linalg::Backend::EIGEN3>(bvec, bvec);

//	for (int32_t i=0; i<avec.size(); i++)
//		result+=CMath::sq(avec[i] - bvec[i]);

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

//	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a);
//	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b);
		
	if (disable_sqrt)
		return result;

	return CMath::sqrt(result);
}
void CEuclideanDistance::precompute_rhs_squared_norms()
{
	m_rhs_sq_norms=true;
	SGVector<float64_t>rhs_sq=SGVector<float64_t>(rhs->get_num_vectors());
	
	for(int32_t idx_i =0; idx_i<rhs->get_num_vectors(); idx_i++)
	{
		SGVector<float64_t> tempvec=((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_i);
		float64_t temp=linalg::dot(tempvec, tempvec);
		rhs_sq[idx_i]=temp;
		((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(tempvec, idx_i);
		
	}
	m_rhs_squared_norms=rhs_sq;

}
void CEuclideanDistance::precompute_lhs_squared_norms()
{
	m_lhs_sq_norms=true;
	SGVector<float64_t>lhs_sq=SGVector<float64_t>(lhs->get_num_vectors());

	for(int32_t idx_i=0; idx_i<lhs->get_num_vectors(); idx_i++)
	{
		SGVector<float64_t> tempvec=((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_i);
		float64_t temp=linalg::dot(tempvec, tempvec);
		lhs_sq[idx_i]=temp;
		((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(tempvec, idx_i);
	}

	m_lhs_squared_norms=lhs_sq;
}

void CEuclideanDistance::set_dot_enabled(bool dot)
{
	m_dot_enabled=dot;
}

bool CEuclideanDistance::get_dot_enabled()
{
	return m_dot_enabled;
}

void CEuclideanDistance::reset_rhs_squared_norms()
{
	m_rhs_sq_norms=false;
}
void CEuclideanDistance::reset_lhs_squared_norms()
{
	m_lhs_sq_norms=false;
}

void CEuclideanDistance::init()
{
	disable_sqrt=false;
	m_rhs_sq_norms=false;
	m_lhs_sq_norms=false;
	m_dot_enabled=false;
//	m_rhs_squared_norms.zero();
//	m_lhs_squared_norms.zero();
	m_parameters->add(&disable_sqrt, "disable_sqrt", "If sqrt shall not be applied.");
}

float64_t CEuclideanDistance::distance_upper_bounded(int32_t idx_a, int32_t idx_b, float64_t upper_bound)
{
	int32_t alen, blen;
	bool afree, bfree;
	float64_t result=0;

	upper_bound *= upper_bound;

	float64_t* avec=((CDenseFeatures<float64_t>*) lhs)->
		get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=((CDenseFeatures<float64_t>*) rhs)->
		get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	for (int32_t i=0; i<alen; i++)
	{
		result+=CMath::sq(avec[i] - bvec[i]);

		if (result > upper_bound)
		{
			((CDenseFeatures<float64_t>*) lhs)->
				free_feature_vector(avec, idx_a, afree);
			((CDenseFeatures<float64_t>*) rhs)->
				free_feature_vector(bvec, idx_b, bfree);

			if (disable_sqrt)
				return result;
			else
				return CMath::sqrt(result);
		}
	}

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	if (disable_sqrt)
		return result;
	else
		return CMath::sqrt(result);
}

SGMatrix<float64_t> CEuclideanDistance::get_precomputed_distance()
{
	SGMatrix<float64_t>prod(rhs->get_num_vectors(), lhs->get_num_vectors());
	



/*
	
	SGMatrix<float64_t>lhs_m = ((CDenseFeatures<float64_t>*) lhs)->get_feature_matrix();
	SGMatrix<float64_t>rhs_m = ((CDenseFeatures<float64_t>*) rhs)->get_feature_matrix();
	SGMatrix<float64_t>prod(rhs->get_num_vectors(), lhs->get_num_vectors());
   	linalg::matrix_product(rhs_m, lhs_m, prod, true);
	linalg::scale(prod, prod, -2.0);

	SGMatrix<float64_t>sq_r = SGMatrix<float64_t>(((CDenseFeatures<float64_t>*) rhs)->get_num_features(), ((CDenseFeatures<float64_t>*) rhs)->get_num_vectors());
	SGMatrix<float64_t>sq_l = SGMatrix<float64_t>(((CDenseFeatures<float64_t>*) lhs)->get_num_features(), ((CDenseFeatures<float64_t>*) lhs)->get_num_vectors());

	sq_r = linalg::elementwise_square(rhs_m);
	sq_l = linalg::elementwise_square(lhs_m);

	SGVector<float64_t>r_norm(rhs->get_num_vectors());
	linalg::colwise_sum(sq_r, r_norm);
	SGVector<float64_t>l_norm(lhs->get_num_vectors());
	linalg::colwise_sum(sq_l, l_norm);

	SGMatrix<float64_t>ln(rhs->get_num_vectors(), lhs->get_num_vectors());
	SGMatrix<float64_t>rn(rhs->get_num_vectors(), lhs->get_num_vectors());
	linalg::set_rows_const(ln, l_norm);
	linalg::set_rows_const(rn, r_norm);

	linalg::add(prod, ln, prod);
	linalg::add(prod, rn, prod);*/
	return prod;
}
 
SGVector<float64_t> CEuclideanDistance::multiple_distance(int32_t idx_a)
{
	
	SGVector<float64_t> avec=((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a);
	SGMatrix<float64_t>rhs_m = ((CDenseFeatures<float64_t>*) rhs)->get_feature_matrix();
	SGVector<float64_t>out = SGVector<float64_t>(rhs->get_num_vectors());
//	SG_SPRINT("here %d %d %d", rhs_m.num_cols, rhs_m.num_rows, avec.size())//.size())
	linalg::apply(rhs_m, avec, out, true);
//	SG_SPRINT("here %d", out.size())
	linalg::scale(out, out, -2.0);
	SGVector<float64_t> result = SGVector<float64_t>(rhs->get_num_vectors());

//	SGMatrix<float64_t>sq_r = SGMatrix<float64_t>(((CDenseFeatures<float64_t>*) rhs)->get_num_features(), ((CDenseFeatures<float64_t>*) rhs)->get_num_vectors());
	
//	SGMatrix<float64_t>sq_r = linalg::elementwise_square(rhs_m);
	SGVector<float64_t>r_norm(rhs->get_num_vectors());
//	linalg::colwise_sum(sq_r, r_norm);

	for(int32_t idx_i =0; idx_i<rhs->get_num_vectors(); idx_i++)
	{
		SGVector<float64_t> bvec=((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_i);
		float64_t temp = linalg::dot(bvec, bvec);
		r_norm[idx_i] = temp;
		((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_i);
		
	}

	linalg::add(out, r_norm, result);
	float64_t l_n;
	if (m_lhs_sq_norms)
		l_n = m_lhs_squared_norms[idx_a];
	else
		l_n = linalg::dot(avec, avec);
	SGVector<float64_t> l_norm = SGVector<float64_t>(rhs->get_num_vectors());
	l_norm.set_const(l_n);

	SGVector<float64_t> dist = SGVector<float64_t>(rhs->get_num_vectors());
	linalg::add(result, l_norm, dist);	

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a);
	return dist;
}
	

