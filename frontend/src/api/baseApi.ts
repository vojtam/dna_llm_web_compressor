import axios, {AxiosRequestConfig } from 'axios';
import { ApiResponseMultiple, ApiResponseMultiplePaginated, ApiResponseSingle } from "./types.ts";
export const BASE_API_URL = "http://localhost:8000/";

const axiosInstance = axios.create({
    baseURL: BASE_API_URL
})

async function getSingle<T>(url: string, config?: AxiosRequestConfig) {
    return await axiosInstance.get<ApiResponseSingle<T>>(url, config);
}
async function getAll<T>(url: string, config?: AxiosRequestConfig) {
    const response = await axiosInstance.get<ApiResponseMultiple<T>>(url, config);
    return response.data;
}
async function getAllPaginated<T>(url: string, config?: AxiosRequestConfig) {
    return await axiosInstance.get<ApiResponseMultiplePaginated<T>>(url, config);
}

async function deleteSingle<T>(url: string, config?: AxiosRequestConfig) {
    return await axiosInstance.delete<ApiResponseSingle<T>>(url, config);
}

async function postSingle<T>(url: string, data: any, config?: AxiosRequestConfig) {
    console.log(url)
    return await axiosInstance.post<ApiResponseSingle<T>>(url, data, config);
}

async function putSingle<T>(url: string, data: any, config?: AxiosRequestConfig) {
    return await axiosInstance.put<ApiResponseSingle<T>>(url, data, config);
}

const BaseApi = {
    getSingle,
    getAll,
    getAllPaginated,
    deleteSingle,
    postSingle,
    putSingle,
};

export default BaseApi;

