export type ApiResponse = {
    verbose?: string;
    messageType?: 'success' | 'error' | 'info' | 'warning';
};

export type ApiResponseSingle<T> = ApiResponse & {
    data: T;
};

export type ApiResponseMultiple<T> = ApiResponse & {
    data: T[];
};

export type ApiResponseMultiplePaginated<T> = ApiResponseMultiple<T> & {
    pagination: {
        totalPages: number;
        currentPage: number;
    };
};

