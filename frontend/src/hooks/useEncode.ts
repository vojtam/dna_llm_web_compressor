import { useMutation } from '@tanstack/react-query';
import BaseApi from '../api/baseApi';

export const useEncode = () => {
    const encodeSeq = async (locus: string) => {
        const response = await BaseApi.postSingle('/encode', {locus: locus});
        if (response.status !== 200) {
            return null;
        }
        console.log(response)
        return response
    }
    const mutation = useMutation({
        mutationFn: encodeSeq
    })

    return mutation
}