#ifndef CUSTOMTYPE_H
#define CUSTOMTYPE_H

enum CROSS_VALIDATION_SYMBOL { STAGE_ONE, STAGE_TWO, STAGE_FULL };


class MySVM : public CvSVM
{
public:
    double * get_alpha_vector()
    {
        return this->decision_func->alpha;
    }

    float get_rho()
    {
        return this->decision_func->rho;
    }

    std::vector<float> get_primal_form() const
    {
      std::vector<float> support_vector;

      int sv_count = get_support_vector_count();

      const CvSVMDecisionFunc* df = decision_func;
      const double* alphas = df[0].alpha;
      double rho = df[0].rho;
      int var_count = get_var_count();

      support_vector.resize(var_count, 0);

      for (unsigned int r = 0; r < (unsigned)sv_count; r++)
      {
        float myalpha = alphas[r];
        const float* v = get_support_vector(r);
        for (int j = 0; j < var_count; j++,v++)
        {
          support_vector[j] += (-myalpha) * (*v);
        }
      }

      support_vector.push_back(rho);

      return support_vector;
    }

};

#endif // CUSTOMTYPE_H
