import numpy
import itertools


def e(*args):
    return numpy.einsum(*args, optimize=True)


def p_count(permutation):
    visited = [False] * len(permutation)
    result = 0
    for i in range(len(permutation)):
        if not visited[i]:
            j = i
            while permutation[j] != i:
                j = permutation[j]
                result += 1
                visited[j] = True
    return result


def p(spec, tensor):
    # print("[{}]".format(spec))
    result = tensor.copy()

    perm_mask = list(i != '.' for i in spec)

    all_indexes = numpy.arange(len(spec))
    perm_indexes = all_indexes[perm_mask]
    
    included = set()
    base_spec = spec.translate(None, ".")

    for i, order in enumerate(itertools.permutations(range(len(spec) - spec.count('.')))):
        this_spec = ''.join(base_spec[_i] for _i in order)
        if i > 0 and this_spec not in included:
            dims = all_indexes.copy()
            dims[perm_mask] = perm_indexes[list(order)]
            perm_tensor = numpy.transpose(tensor, dims)

            if p_count(order) % 2 == 0:
                # print("  + {} {}".format(this_spec, repr(dims)))
                result += perm_tensor
            else:
                # print("  - {} {}".format(this_spec, repr(dims)))
                result -= perm_tensor
        included.add(this_spec)
    return result


def equations_s(oo, ov, vo, vv, oovo, oovv, ovvo, ovvv, t1):
    hE = scalar = 0
    hE += e("ab,ib->ia", vv, t1)  # d1_ov
    hE += e("ai->ia", vo)  # d2_ov
    hE -= e("ia,ja,ib->jb", ov, t1, t1)  # d5_ov
    scalar += e("ia,ia", ov, t1)  # s7
    hE -= e("ji,ja->ia", oo, t1)  # d9_ov
    hE -= e("icba,ia,jb->jc", ovvv, t1, t1)  # d12_ov
    hE += e("jabi,jb->ia", ovvo, t1)  # d15_ov
    hE += e("ijcb,ia,kb,jc->ka", oovv, t1, t1, t1)  # d19_ov
    scalar += 1./2 * e("jiba,ia,jb", oovv, t1, t1)  # s24
    hE += e("ikbj,ia,kb->ja", oovo, t1, t1)  # d27_ov
    return hE, scalar


def equations_sd(oo, ov, vo, vv, oooo, oovo, oovv, ovoo, ovvo, ovvv, vvoo, vvvo, vvvv, t1, t2):
    hE = hhEE = scalar = 0
    hE += e("ab,ib->ia", vv, t1)  # d1_ov
    hhEE += p("..ab", e("cb,ijba->ijca", vv, t2))  # d2_oovv
    hE += e("ai->ia", vo)  # d3_ov
    hE -= e("ia,ja,ib->jb", ov, t1, t1)  # d6_ov
    scalar += e("ia,ia", ov, t1)  # s8
    hhEE -= p("..ab", e("kb,kc,ijba->ijca", ov, t1, t2))  # d10_oovv
    hhEE -= p("ab..", e("ia,ka,ijcb->jkbc", ov, t1, t2))  # d12_oovv
    hE += e("ia,ijab->jb", ov, t2)  # d13_ov
    hE -= e("ij,ia->ja", oo, t1)  # d15_ov
    hhEE -= p("ab..", e("ji,jkba->ikba", oo, t2))  # d16_oovv
    hhEE += 1./2 * p("ab..", e("cbad,ia,jd->ijcb", vvvv, t1, t1))  # d19_oovv
    hhEE += 1./2 * e("badc,ijdc->ijba", vvvv, t2)  # d20_oovv
    hhEE += p("ab..", e("baci,jc->ijab", vvvo, t1))  # d22_oovv
    hhEE += e("baij->ijba", vvoo)  # d23_oovv
    hhEE -= 1./2 * p("ab..", p("..ab", e("iabc,jb,kc,id->jkda", ovvv, t1, t1, t1)))  # d27_oovv
    hE -= e("jbac,ia,jc->ib", ovvv, t1, t1)  # d29_ov
    hhEE -= p("..ab", e("iacb,ib,jkcd->jkad", ovvv, t1, t2))  # d33_oovv
    hhEE -= 1./2 * p("..ab", e("idcb,ia,jkcb->jkad", ovvv, t1, t2))  # d35_oovv
    hhEE -= p("ab..", p("..ab", e("kdac,ia,kjcb->ijdb", ovvv, t1, t2)))  # d37_oovv
    hE += 1./2 * e("jcba,jiba->ic", ovvv, t2)  # d38_ov
    hhEE -= p("ab..", p("..ab", e("kbaj,ia,kc->ijcb", ovvo, t1, t1)))  # d41_oovv
    hE += e("jabi,jb->ia", ovvo, t1)  # d43_ov
    hhEE += p("ab..", p("..ab", e("jabi,jkbc->ikac", ovvo, t2)))  # d44_oovv
    hhEE -= p("..ab", e("kaij,kb->ijba", ovoo, t1))  # d46_oovv
    hhEE += 1./4 * p("ab..", p("..ab", e("jiba,ic,kb,jd,la->kldc", oovv, t1, t1, t1, t1)))  # d51_oovv
    hE -= e("jiba,ia,kb,jc->kc", oovv, t1, t1, t1)  # d54_ov
    scalar += 1./2 * e("jiba,ia,jb", oovv, t1, t1)  # s60
    hhEE -= p("..ab", e("klcb,kc,ld,ijba->ijda", oovv, t1, t1, t2))  # d63_oovv
    hhEE -= 1./2 * p("..ab", e("kldb,ijba,kldc->ijca", oovv, t2, t2))  # d65_oovv
    hhEE += 1./4 * p("..ab", e("ijdc,ia,jb,kldc->klab", oovv, t1, t1, t2))  # d68_oovv
    hhEE += 1./4 * e("ijdc,ijba,kldc->klba", oovv, t2, t2)  # d69_oovv
    hhEE -= p("ab..", e("lkda,ia,ld,kjcb->ijcb", oovv, t1, t1, t2))  # d72_oovv
    hhEE += 1./2 * p("ab..", e("ljba,jiba,lkdc->ikdc", oovv, t2, t2))  # d74_oovv
    hhEE -= p("ab..", p("..ab", e("kjcb,lc,kd,jiba->ilad", oovv, t1, t1, t2)))  # d77_oovv
    hE += e("ijab,ia,jkbc->kc", oovv, t1, t2)  # d79_ov
    hhEE += 1./2 * p("ab..", p("..ab", e("jkbc,jiba,klcd->ilad", oovv, t2, t2)))  # d80_oovv
    hE += 1./2 * e("kjba,kc,jiba->ic", oovv, t1, t2)  # d82_ov
    hhEE += 1./4 * p("ab..", e("ijdc,kc,ld,ijba->klab", oovv, t1, t1, t2))  # d85_oovv
    hE += 1./2 * e("jkac,ia,jkcb->ib", oovv, t1, t2)  # d87_ov
    scalar += 1./4 * e("ijba,ijba", oovv, t2)  # s88
    hhEE += 1./2 * p("ab..", p("..ab", e("jkai,kb,la,jc->ilbc", oovo, t1, t1, t1)))  # d92_oovv
    hE += e("ikbj,ia,kb->ja", oovo, t1, t1)  # d95_ov
    hhEE -= p("ab..", e("kjcl,kc,jiba->ilab", oovo, t1, t2))  # d98_oovv
    hhEE += p("ab..", p("..ab", e("ikcl,ia,kjcb->jlba", oovo, t1, t2)))  # d100_oovv
    hhEE += 1./2 * p("ab..", e("ijcl,kc,ijba->klba", oovo, t1, t2))  # d102_oovv
    hE -= 1./2 * e("ijbk,ijba->ka", oovo, t2)  # d103_ov
    hhEE += 1./2 * p("..ab", e("jikl,ia,jb->klba", oooo, t1, t1))  # d106_oovv
    hhEE += 1./2 * e("ijkl,ijba->klba", oooo, t2)  # d107_oovv
    return hE, hhEE, scalar


def equations_sdt(oo, ov, vo, vv, oooo, oovo, oovv, ovoo, ovvo, ovvv, vvoo, vvvo, vvvv, t1, t2, t3):
    hE = hhEE = hhhEEE = scalar = 0
    hE += e("ab,ib->ia", vv, t1)  # d1_ov
    hhEE += p("..ab", e("ab,ijbc->ijac", vv, t2))  # d2_oovv
    hhhEEE += p("...aba", e("ab,jkibdc->ijkcad", vv, t3))  # d3_ooovvv
    hE += e("ai->ia", vo)  # d4_ov
    hE -= e("ja,ia,jb->ib", ov, t1, t1)  # d7_ov
    scalar += e("ia,ia", ov, t1)  # s9
    hhEE -= p("..ab", e("ia,ic,jkab->jkcb", ov, t1, t2))  # d11_oovv
    hhhEEE -= p("aab...", p("...aba", e("ia,jkab,ildc->jkldbc", ov, t2, t2)))  # d12_ooovvv
    hhEE -= p("ab..", e("jc,kc,jiba->ikab", ov, t1, t2))  # d14_oovv
    hE += e("jb,jiba->ia", ov, t2)  # d15_ov
    hhhEEE -= p("...aba", e("ia,id,kljacb->jklbdc", ov, t1, t3))  # d17_ooovvv
    hhhEEE -= p("aab...", e("ia,la,ikjdcb->jklbcd", ov, t1, t3))  # d19_ooovvv
    hhEE += e("kc,kjicba->ijab", ov, t3)  # d20_oovv
    hE -= e("ji,ja->ia", oo, t1)  # d22_ov
    hhEE -= p("ab..", e("ji,jkba->ikba", oo, t2))  # d23_oovv
    hhhEEE -= p("abb...", e("ji,jlkcba->iklcab", oo, t3))  # d24_ooovvv
    hhEE += 1./2 * p("ab..", e("dcab,ia,jb->ijdc", vvvv, t1, t1))  # d27_oovv
    hhhEEE += p("aab...", p("...aba", e("edbc,kc,ijba->ijkead", vvvv, t1, t2)))  # d29_ooovvv
    hhEE += 1./2 * e("dcba,ijba->ijdc", vvvv, t2)  # d30_oovv
    hhhEEE += 1./2 * p("...abb", e("edcb,jkicba->ijkaed", vvvv, t3))  # d31_ooovvv
    hhEE += p("ab..", e("baci,jc->ijab", vvvo, t1))  # d33_oovv
    hhhEEE += p("abb...", p("...aab", e("baci,jkcd->ijkabd", vvvo, t2)))  # d34_ooovvv
    hhEE += e("baij->ijba", vvoo)  # d35_oovv
    hhEE -= 1./2 * p("ab..", p("..ab", e("iabc,jc,id,kb->jkad", ovvv, t1, t1, t1)))  # d39_oovv
    hE += e("jcba,ia,jb->ic", ovvv, t1, t1)  # d41_ov
    hhhEEE -= p("aab...", p("...abc", e("iabc,id,lb,jkce->jklaed", ovvv, t1, t1, t2)))  # d46_ooovvv
    hhEE += p("..ab", e("idac,ia,jkcb->jkdb", ovvv, t1, t2))  # d48_oovv
    hhhEEE += p("aab...", p("...abc", e("ledb,ijba,lkdc->ijkeac", ovvv, t2, t2)))  # d49_ooovvv
    hhEE -= 1./2 * p("..ab", e("kcba,kd,ijba->ijdc", ovvv, t1, t2))  # d51_oovv
    hhhEEE -= 1./2 * p("abb...", p("...aab", e("jedc,jiba,kldc->iklabe", ovvv, t2, t2)))  # d52_ooovvv
    hhhEEE -= 1./2 * p("abc...", p("...abb", e("iacb,jb,lc,iked->jklade", ovvv, t1, t1, t2)))  # d55_ooovvv
    hhEE += p("ab..", p("..ab", e("iabc,kc,ijbd->jkda", ovvv, t1, t2)))  # d57_oovv
    hE += 1./2 * e("jcba,jiba->ic", ovvv, t2)  # d58_ov
    hhhEEE += p("...aba", e("ledc,ld,jkicba->ijkaeb", ovvv, t1, t3))  # d60_ooovvv
    hhhEEE -= 1./2 * p("...abc", e("iedc,ia,kljdcb->jklbae", ovvv, t1, t3))  # d62_ooovvv
    hhhEEE -= p("aab...", p("...aab", e("kdec,le,kjicba->ijlabd", ovvv, t1, t3)))  # d64_ooovvv
    hhEE += 1./2 * p("..ab", e("kdcb,kjicba->ijad", ovvv, t3))  # d65_oovv
    hhEE -= p("ab..", p("..ab", e("jabi,kb,jc->ikac", ovvo, t1, t1)))  # d68_oovv
    hE += e("jabi,jb->ia", ovvo, t1)  # d70_ov
    hhhEEE -= p("abb...", p("...abc", e("jabi,jd,klbc->ikladc", ovvo, t1, t2)))  # d72_ooovvv
    hhhEEE -= p("abc...", p("...aba", e("jcdk,ld,jiba->iklacb", ovvo, t1, t2)))  # d74_ooovvv
    hhEE += p("ab..", p("..ab", e("jabi,jkbc->ikac", ovvo, t2)))  # d75_oovv
    hhhEEE += p("aab...", p("...aab", e("kdcl,kjicba->ijlabd", ovvo, t3)))  # d76_ooovvv
    hhEE -= p("..ab", e("kaij,kb->ijba", ovoo, t1))  # d78_oovv
    hhhEEE -= p("abb...", p("...aab", e("jckl,jiba->iklabc", ovoo, t2)))  # d79_ooovvv
    hhEE += 1./4 * p("ab..", p("..ab", e("jidc,ia,jb,kc,ld->klab", oovv, t1, t1, t1, t1)))  # d84_oovv
    hE -= e("ijcb,ia,jb,kc->ka", oovv, t1, t1, t1)  # d87_ov
    scalar += 1./2 * e("ijab,ia,jb", oovv, t1, t1)  # s93
    hhhEEE += 1./2 * p("abb...", p("...abc", e("kjac,ia,jb,ke,lmcd->ilmebd", oovv, t1, t1, t1, t2)))  # d97_ooovvv
    hhEE -= p("..ab", e("ijac,ia,jb,klcd->klbd", oovv, t1, t1, t2))  # d100_oovv
    hhhEEE -= p("aab...", p("...aba", e("lkbc,kc,ijba,lmed->ijmead", oovv, t1, t2, t2)))  # d103_ooovvv
    hhhEEE -= p("abb...", p("...abc", e("jiab,ic,jkad,lmbe->klmdce", oovv, t1, t2, t2)))  # d105_ooovvv
    hhhEEE += 1./2 * p("abb...", p("...aab", e("jkba,ia,jkdc,lmbe->ilmcde", oovv, t1, t2, t2)))  # d107_ooovvv
    hhEE -= 1./2 * p("..ab", e("kldb,ijba,kldc->ijca", oovv, t2, t2))  # d108_oovv
    hhEE += 1./4 * p("..ab", e("ijdc,ia,jb,kldc->klab", oovv, t1, t1, t2))  # d111_oovv
    hhhEEE += 1./2 * p("abb...", p("...aab", e("jmdc,me,jiba,kldc->iklabe", oovv, t1, t2, t2)))  # d113_ooovvv
    hhEE += 1./4 * e("ijdc,ijba,kldc->klba", oovv, t2, t2)  # d114_oovv
    hhhEEE += 1./2 * p("abc...", p("...abb", e("jiba,ka,mb,ie,jldc->klmecd", oovv, t1, t1, t1, t2)))  # d118_ooovvv
    hhEE += p("ab..", e("kjcd,kd,lc,jiba->ilab", oovv, t1, t1, t2))  # d120_oovv
    hhhEEE -= p("abc...", p("...aba", e("kjcd,md,jiba,klce->ilmaeb", oovv, t1, t2, t2)))  # d124_ooovvv
    hhEE -= 1./2 * p("ab..", e("jkba,jiba,kldc->ildc", oovv, t2, t2))  # d125_oovv
    hhEE += p("ab..", p("..ab", e("jkdb,kc,ld,jiba->ilac", oovv, t1, t1, t2)))  # d128_oovv
    hE += e("ijab,ia,jkbc->kc", oovv, t1, t2)  # d130_ov
    hhEE += 1./2 * p("ab..", p("..ab", e("ijab,ikac,jlbd->klcd", oovv, t2, t2)))  # d131_oovv
    hE += 1./2 * e("kjba,kc,jiba->ic", oovv, t1, t2)  # d133_ov
    hhEE += 1./4 * p("ab..", e("klab,ia,jb,kldc->ijdc", oovv, t1, t1, t2))  # d136_oovv
    hE += 1./2 * e("ijcb,kc,ijba->ka", oovv, t1, t2)  # d138_ov
    scalar += 1./4 * e("ijba,ijba", oovv, t2)  # s139
    hhhEEE -= p("...aba", e("imad,ia,me,kljdcb->jklbec", oovv, t1, t1, t3))  # d142_ooovvv
    hhhEEE += 1./2 * p("...aba", e("ijcb,ijba,lmkced->klmdae", oovv, t2, t3))  # d144_ooovvv
    hhhEEE += 1./4 * p("...abc", e("ijcb,ia,je,lmkcbd->klmdae", oovv, t1, t1, t3))  # d147_ooovvv
    hhhEEE += 1./4 * p("...abb", e("ijdc,ijba,lmkdce->klmeba", oovv, t2, t3))  # d148_ooovvv
    hhhEEE -= p("aab...", e("ijab,jb,ma,ilkedc->klmcde", oovv, t1, t1, t3))  # d151_ooovvv
    hhhEEE -= 1./2 * p("aab...", e("ijba,imba,jlkedc->klmcde", oovv, t2, t3))  # d153_ooovvv
    hhhEEE -= p("aab...", p("...aab", e("lkdc,md,le,kjicba->ijmabe", oovv, t1, t1, t3)))  # d156_ooovvv
    hhEE += e("ilad,ia,lkjdcb->jkbc", oovv, t1, t3)  # d158_oovv
    hhhEEE += p("abb...", p("...abb", e("jmbe,jiba,mlkedc->iklacd", oovv, t2, t3)))  # d159_ooovvv
    hhEE += 1./2 * p("..ab", e("lkcb,ld,kjicba->ijad", oovv, t1, t3))  # d161_oovv
    hhhEEE += 1./2 * p("abb...", p("...aba", e("jmed,jiba,mlkedc->iklacb", oovv, t2, t3)))  # d162_ooovvv
    hhhEEE += 1./4 * p("abc...", e("ijab,la,mb,ijkedc->klmced", oovv, t1, t1, t3))  # d165_ooovvv
    hhhEEE += 1./4 * p("aab...", e("ijba,klba,ijmedc->klmedc", oovv, t2, t3))  # d166_ooovvv
    hhEE += 1./2 * p("ab..", e("ijab,ka,ijlbdc->kldc", oovv, t1, t3))  # d168_oovv
    hhhEEE += 1./2 * p("abb...", p("...aab", e("jkdc,lmde,jkicba->ilmabe", oovv, t2, t3)))  # d169_ooovvv
    hE += 1./4 * e("ijba,ijkbac->kc", oovv, t3)  # d170_ov
    hhEE += 1./2 * p("ab..", p("..ab", e("jick,ia,jb,lc->klab", oovo, t1, t1, t1)))  # d174_oovv
    hE -= e("jkai,ja,kb->ib", oovo, t1, t1)  # d177_ov
    hhhEEE += 1./2 * p("abb...", p("...abc", e("jkai,kc,jd,lmab->ilmcdb", oovo, t1, t1, t2)))  # d181_ooovvv
    hhhEEE += 1./2 * p("aab...", p("...aba", e("ijdm,ijba,kldc->klmbca", oovo, t2, t2)))  # d182_ooovvv
    hhhEEE += p("abc...", p("...aab", e("kjai,ma,kd,jlcb->ilmcbd", oovo, t1, t1, t2)))  # d185_ooovvv
    hhEE += p("ab..", e("jkai,ka,jlcb->ilcb", oovo, t1, t2))  # d187_oovv
    hhhEEE += p("abc...", p("...aba", e("jldm,jiba,lkdc->ikmacb", oovo, t2, t2)))  # d188_ooovvv
    hhEE += p("ab..", p("..ab", e("kjai,kb,jlac->ilbc", oovo, t1, t2)))  # d190_oovv
    hhEE += 1./2 * p("ab..", e("klaj,ia,klcb->ijcb", oovo, t1, t2))  # d192_oovv
    hE -= 1./2 * e("ijbk,ijba->ka", oovo, t2)  # d193_ov
    hhhEEE -= p("abb...", e("ikaj,ia,kmldcb->jlmdbc", oovo, t1, t3))  # d195_ooovvv
    hhhEEE += p("aab...", p("...aab", e("ildm,ia,lkjdcb->jkmbca", oovo, t1, t3)))  # d197_ooovvv
    hhhEEE += 1./2 * p("abc...", e("klam,ia,kljdcb->ijmdbc", oovo, t1, t3))  # d199_ooovvv
    hhEE -= 1./2 * p("ab..", e("jkai,jklacb->ilcb", oovo, t3))  # d200_oovv
    hhEE += 1./2 * p("..ab", e("iljk,ia,lb->jkab", oooo, t1, t1))  # d203_oovv
    hhhEEE += p("aab...", p("...abb", e("lkij,la,kmcb->ijmacb", oooo, t1, t2)))  # d205_ooovvv
    hhEE += 1./2 * e("klij,klba->ijba", oooo, t2)  # d206_oovv
    hhhEEE += 1./2 * p("aab...", e("klij,klmcba->ijmcba", oooo, t3))  # d207_ooovvv
    return hE, hhEE, hhhEEE, scalar
